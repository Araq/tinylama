## Generalized LLM forward pass using Arraymancer.

import std/[math]
import arraymancer
import ./tensor
import ./model

proc getTensorOr(m: var Model, a, b: string): GGTensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

proc embeddingLookup(weight: GGTensor, tokens: seq[int32], nVocab, nEmb: int): GGTensor =
  result = newGGTensor(@[nEmb, tokens.len])
  let wShape = weight.shape
  let embDimIdx = if wShape.len > 1 and wShape[1] == nEmb: 1 else: 0
  for i, t in tokens:
    let tid = int(t)
    if embDimIdx == 1:
      result.at[_, i] = weight.at[tid, _].reshape(nEmb, 1)
    else:
      result.at[_, i] = weight.at[_, tid].reshape(nEmb, 1)

proc linearGGMLCol(x: GGTensor, w: GGTensor): GGTensor =
  result = matmul(w, x)

type
  KvCache* = object
    k*: seq[GGTensor]
    v*: seq[GGTensor]
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int

proc initKvCache*(hp: HParams, maxLen: int): KvCache =
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = hp.nHeadKv
  result.headDim = hp.nEmb div hp.nHead
  let kvDim = hp.nHeadKv * result.headDim
  result.k = newSeq[GGTensor](hp.nLayer)
  result.v = newSeq[GGTensor](hp.nLayer)
  for i in 0 ..< hp.nLayer:
    result.k[i] = newGGTensor(@[kvDim, maxLen])
    result.v[i] = newGGTensor(@[kvDim, maxLen])

proc applyRopeInternal(x: var GGTensor, nHead, headDim, ropeDim: int, base: float32, startPos: int) =
  let xShape = x.shape
  let seqLen = xShape[1]
  var cpuX = x.at.toCpu()
  for h in 0 ..< nHead:
    for p in 0 ..< seqLen:
      let pos = float32(startPos + p)
      for i in 0 ..< ropeDim div 2:
        let theta = pow(base, -float32(2 * i) / float32(ropeDim))
        let angle = pos * theta
        let c = cos(angle)
        let s = sin(angle)
        let idx0 = h * headDim + 2 * i
        let idx1 = h * headDim + 2 * i + 1
        let v0 = cpuX[idx0, p]
        let v1 = cpuX[idx1, p]
        cpuX[idx0, p] = v0 * c - v1 * s
        cpuX[idx1, p] = v0 * s + v1 * c
  x.at = cpuX.toDevice()

proc attentionFull(q, k, v, wo: GGTensor, nHead, nHeadKv, headDim: int): GGTensor =
  let seqLen = q.at.shape[1]
  let group = nHead div nHeadKv

  let qr = q.at.reshape(nHead, headDim, seqLen)
  let kr = k.at.reshape(nHeadKv, headDim, seqLen)
  let vr = v.at.reshape(nHeadKv, headDim, seqLen)

  # Moving to CPU for complex attention logic (masking + per-head)
  var outAttnCpu = newTensor[float32](nHead * headDim, seqLen)
  let qrCpu = qr.toCpu()
  let krCpu = kr.toCpu()
  let vrCpu = vr.toCpu()

  for h in 0 ..< nHead:
    let kvh = h div group
    let qh = qrCpu[h, _, _].reshape(headDim, seqLen)
    let kh = krCpu[kvh, _, _].reshape(headDim, seqLen)
    let vh = vrCpu[kvh, _, _].reshape(headDim, seqLen)

    # scores = qh.T * kh / sqrt(headDim)
    var scores = (qh.transpose() * kh) / sqrt(float32(headDim))

    # Causal Mask
    for i in 0 ..< seqLen:
      for j in i + 1 ..< seqLen:
        scores[i, j] = -1e30'f32

    let scoresNorm = scores.softmax()
    let res = vh * scoresNorm.transpose()
    outAttnCpu[h * headDim ..< (h+1) * headDim, _] = res

  result = matmul(wo, outAttnCpu.toDevice().toGGTensor())

proc attentionCached(q: GGTensor, kCache, vCache: GGTensor, curLen, nHead, nHeadKv, headDim: int): GGTensor =
  let group = nHead div nHeadKv
  let qr = q.at.reshape(nHead, headDim, 1)
  let kActive = kCache.at[_, 0 ..< curLen]
  let vActive = vCache.at[_, 0 ..< curLen]
  let kr = kActive.reshape(nHeadKv, headDim, curLen)
  let vr = vActive.reshape(nHeadKv, headDim, curLen)

  var outAttnCpu = newTensor[float32](nHead * headDim, 1)
  let qrCpu = qr.toCpu()
  let krCpu = kr.toCpu()
  let vrCpu = vr.toCpu()

  for h in 0 ..< nHead:
    let kvh = h div group
    let qh = qrCpu[h, _, 0].reshape(headDim, 1)
    let kh = krCpu[kvh, _, _].reshape(headDim, curLen)
    let vh = vrCpu[kvh, _, _].reshape(headDim, curLen)

    let scores = (qh.transpose() * kh) / sqrt(float32(headDim))
    let scoresNorm = scores.softmax()
    let res = vh * scoresNorm.transpose()
    outAttnCpu[h * headDim ..< (h+1) * headDim, _] = res
  result = outAttnCpu.toDevice().toGGTensor()

proc ffn(x: GGTensor, wGate, wUp, wDown: GGTensor, actType: string): GGTensor =
  let gate = linearGGMLCol(x, wGate)
  let up = linearGGMLCol(x, wUp)
  var act: GGTensor
  if actType == "gelu":
    act = mul(gelu(gate), up)
  else:
    act = mul(silu(gate), up)
  result = linearGGMLCol(act, wDown)

proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): GGTensor =
  let hp = m.hparams
  let headDim = hp.nEmb div hp.nHead
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let tokEmb = getTensorOr(m, "tok_embeddings.weight", "token_embd.weight")
  var x = embeddingLookup(tokEmb, tokens, hp.nVocab, hp.nEmb)
  cache.curLen = tokens.len
  for layer in 0 ..< hp.nLayer:
    let attnNorm = m.getTensor("blk." & $layer & ".attn_norm.weight")
    let ffnNorm = m.getTensor("blk." & $layer & ".ffn_norm.weight")
    let wq = m.getTensor("blk." & $layer & ".attn_q.weight")
    let wk = m.getTensor("blk." & $layer & ".attn_k.weight")
    let wv = m.getTensor("blk." & $layer & ".attn_v.weight")
    let wo = m.getTensor("blk." & $layer & ".attn_output.weight")
    let wGate = m.getTensor("blk." & $layer & ".ffn_gate.weight")
    let wUp = m.getTensor("blk." & $layer & ".ffn_up.weight")
    let wDown = m.getTensor("blk." & $layer & ".ffn_down.weight")

    let xNorm = if hp.normType == "rms": rmsnormCols(x, attnNorm, hp.rmsEps)
                else: rmsnormCols(x, attnNorm, hp.rmsEps) # Fallback to RMS

    var q = linearGGMLCol(xNorm, wq)
    var k = linearGGMLCol(xNorm, wk)
    let v = linearGGMLCol(xNorm, wv)

    applyRopeInternal(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0)
    applyRopeInternal(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0)

    cache.k[layer].at[_, 0 ..< tokens.len] = k.at
    cache.v[layer].at[_, 0 ..< tokens.len] = v.at

    let attnOut = attentionFull(q, k, v, wo, hp.nHead, hp.nHeadKv, headDim)
    x = add(x, attnOut)

    let xNorm2 = if hp.normType == "rms": rmsnormCols(x, ffnNorm, hp.rmsEps)
                 else: rmsnormCols(x, ffnNorm, hp.rmsEps)

    let ffnOut = ffn(xNorm2, wGate, wUp, wDown, hp.actType)
    x = add(x, ffnOut)

  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, norm, hp.rmsEps)
  result = matmul(outW, xNormFinal)

proc forwardDecode*(m: var Model, token: int32, cache: var KvCache): GGTensor =
  let hp = m.hparams
  let headDim = hp.nEmb div hp.nHead
  let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
  let tokEmb = getTensorOr(m, "tok_embeddings.weight", "token_embd.weight")
  var x = embeddingLookup(tokEmb, @[token], hp.nVocab, hp.nEmb)
  let pos = cache.curLen
  for layer in 0 ..< hp.nLayer:
    let attnNorm = m.getTensor("blk." & $layer & ".attn_norm.weight")
    let ffnNorm = m.getTensor("blk." & $layer & ".ffn_norm.weight")
    let wq = m.getTensor("blk." & $layer & ".attn_q.weight")
    let wk = m.getTensor("blk." & $layer & ".attn_k.weight")
    let wv = m.getTensor("blk." & $layer & ".attn_v.weight")
    let wo = m.getTensor("blk." & $layer & ".attn_output.weight")
    let wGate = m.getTensor("blk." & $layer & ".ffn_gate.weight")
    let wUp = m.getTensor("blk." & $layer & ".ffn_up.weight")
    let wDown = m.getTensor("blk." & $layer & ".ffn_down.weight")

    let xNorm = if hp.normType == "rms": rmsnormCols(x, attnNorm, hp.rmsEps)
                else: rmsnormCols(x, attnNorm, hp.rmsEps)

    var q = linearGGMLCol(xNorm, wq)
    var k = linearGGMLCol(xNorm, wk)
    let v = linearGGMLCol(xNorm, wv)

    applyRopeInternal(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos)
    applyRopeInternal(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos)

    cache.k[layer].at[_, pos] = k.at[_, 0]
    cache.v[layer].at[_, pos] = v.at[_, 0]

    let attnCtx = attentionCached(q, cache.k[layer], cache.v[layer], pos + 1, hp.nHead, hp.nHeadKv, headDim)
    let attnOut = linearGGMLCol(attnCtx, wo)
    x = add(x, attnOut)

    let xNorm2 = if hp.normType == "rms": rmsnormCols(x, ffnNorm, hp.rmsEps)
                 else: rmsnormCols(x, ffnNorm, hp.rmsEps)

    let ffnOut = ffn(xNorm2, wGate, wUp, wDown, hp.actType)
    x = add(x, ffnOut)
  cache.curLen = pos + 1
  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, norm, hp.rmsEps)
  result = matmul(outW, xNormFinal)
