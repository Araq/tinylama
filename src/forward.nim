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

proc applyRopeSingle(x: var GGTensor, nHead, headDim, ropeDim: int, base: float32) =
  ropeInplace(x, base)

proc applyRopeAtPos(x: var GGTensor, nHead, headDim, ropeDim: int, base: float32, pos: int) =
  ropeInplace(x, base, pos)

proc attentionFull(q, k, v, wo: GGTensor, nHead, nHeadKv, headDim: int): GGTensor =
  let seqLen = q.at.shape[1]
  let group = nHead div nHeadKv
  let qr = q.at.reshape(nHead, headDim, seqLen)
  let kr = k.at.reshape(nHeadKv, headDim, seqLen)
  let vr = v.at.reshape(nHeadKv, headDim, seqLen)

  var outAttnCpu = newTensor[float32](nHead * headDim, seqLen)
  for h in 0 ..< nHead:
    let kvh = h div group
    let qh = qr[h, _, _].reshape(headDim, seqLen)
    let kh = kr[kvh, _, _].reshape(headDim, seqLen)
    let vh = vr[kvh, _, _].reshape(headDim, seqLen)

    # We do attention on CPU for now as it's complex to vectorize on GPU without custom kernels
    let qhCpu = qh.toCpu()
    let khCpu = kh.toCpu()
    let vhCpu = vh.toCpu()

    let scores = (qhCpu.transpose() * khCpu) / sqrt(float32(headDim))
    let scoresNorm = scores.softmax()
    let res = vhCpu * scoresNorm.transpose()
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
  for h in 0 ..< nHead:
    let kvh = h div group
    let qh = qr[h, _, 0].reshape(headDim, 1)
    let kh = kr[kvh, _, _].reshape(headDim, curLen)
    let vh = vr[kvh, _, _].reshape(headDim, curLen)

    let qhCpu = qh.toCpu()
    let khCpu = kh.toCpu()
    let vhCpu = vh.toCpu()

    let scores = (qhCpu.transpose() * khCpu) / sqrt(float32(headDim))
    let scoresNorm = scores.softmax()
    let res = vhCpu * scoresNorm.transpose()
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
                else: rmsnormCols(x, attnNorm, hp.rmsEps)

    var q = linearGGMLCol(xNorm, wq)
    var k = linearGGMLCol(xNorm, wk)
    let v = linearGGMLCol(xNorm, wv)

    applyRopeSingle(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase)
    applyRopeSingle(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase)

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

    applyRopeAtPos(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos)
    applyRopeAtPos(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos)

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
