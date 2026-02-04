## Minimal LLaMA-style forward pass (float32, CPU, no KV cache).

import std/[math]
import ./tensor
import ./model

proc getTensorOr(m: var Model, a, b: string): Tensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

proc embeddingLookup(weight: Tensor, tokens: seq[int32], nVocab, nEmb: int): Tensor =
  result = newTensor(@[tokens.len, nEmb])
  if weight.shape.len != 2:
    raise newException(ValueError, "embedding weight must be 2D")
  let a0 = weight.shape[0]
  let a1 = weight.shape[1]
  if a0 == nVocab and a1 == nEmb:
    for i, t in tokens:
      let tid = int(t)
      let row = tid * nEmb
      let outIdx = i * nEmb
      for e in 0 ..< nEmb:
        result.data[outIdx + e] = weight.data[row + e]
  elif a0 == nEmb and a1 == nVocab:
    for i, t in tokens:
      let tid = int(t)
      let outIdx = i * nEmb
      for e in 0 ..< nEmb:
        result.data[outIdx + e] = weight.data[e * nVocab + tid]
  else:
    raise newException(ValueError, "embedding shape mismatch")

type
  KvCache* = object
    k*: seq[Tensor]
    v*: seq[Tensor]
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int

proc initKvCache*(hp: HParams, maxLen: int): KvCache =
  if hp.nHead <= 0:
    raise newException(ValueError, "KV cache requires llama-style head_count")
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = hp.nHeadKv
  result.headDim = hp.nEmb div hp.nHead
  let kvDim = hp.nHeadKv * result.headDim
  result.k = newSeq[Tensor](hp.nLayer)
  result.v = newSeq[Tensor](hp.nLayer)
  for i in 0 ..< hp.nLayer:
    result.k[i] = newTensor(@[maxLen, kvDim])
    result.v[i] = newTensor(@[maxLen, kvDim])

proc applyRopeSingle(x: var Tensor, nHead, headDim, ropeDim: int, base: float32) =
  let seqLen = x.shape[0]
  for h in 0 ..< nHead:
    let hOffset = h * headDim
    for p in 0 ..< seqLen:
      let row = p * x.shape[1]
      for i in 0 ..< ropeDim div 2:
        let idx0 = row + hOffset + 2 * i
        let idx1 = row + hOffset + 2 * i + 1
        let theta = pow(1.0'f32 / base, float32(2 * i) / float32(ropeDim))
        let angle = float32(p) * theta
        let c = cos(angle)
        let s = sin(angle)
        let v0 = x.data[idx0]
        let v1 = x.data[idx1]
        x.data[idx0] = v0 * c - v1 * s
        x.data[idx1] = v0 * s + v1 * c

proc applyRopeAtPos(x: var Tensor, nHead, headDim, ropeDim: int, base: float32, pos: int) =
  let row = 0
  for h in 0 ..< nHead:
    let hOffset = h * headDim
    for i in 0 ..< ropeDim div 2:
      let idx0 = row + hOffset + 2 * i
      let idx1 = row + hOffset + 2 * i + 1
      let theta = pow(1.0'f32 / base, float32(2 * i) / float32(ropeDim))
      let angle = float32(pos) * theta
      let c = cos(angle)
      let s = sin(angle)
      let v0 = x.data[idx0]
      let v1 = x.data[idx1]
      x.data[idx0] = v0 * c - v1 * s
      x.data[idx1] = v0 * s + v1 * c

proc attentionFull(q, k, v, wo: Tensor, nHead, nHeadKv, headDim: int): Tensor =
  let seqLen = q.shape[0]

  var ctx = newTensor(@[seqLen, nHead * headDim])
  let group = nHead div nHeadKv
  for h in 0 ..< nHead:
    let kvh = h div group
    let hOff = h * headDim
    for i in 0 ..< seqLen:
      var scores = newSeq[float32](i + 1)
      for j in 0 ..< i + 1:
        var dot = 0.0'f32
        let qRow = i * q.shape[1] + hOff
        let kRow = j * k.shape[1] + kvh * headDim
        for d in 0 ..< headDim:
          dot += q.data[qRow + d] * k.data[kRow + d]
        scores[j] = dot / sqrt(float32(headDim))
      # softmax
      var maxv = scores[0]
      for j in 1 ..< scores.len:
        if scores[j] > maxv: maxv = scores[j]
      var sum = 0.0'f32
      for j in 0 ..< scores.len:
        scores[j] = exp(scores[j] - maxv)
        sum += scores[j]
      let inv = 1.0'f32 / sum
      for j in 0 ..< scores.len:
        scores[j] *= inv
      # weighted sum of V
      let outRow = i * ctx.shape[1] + hOff
      for d in 0 ..< headDim:
        var acc = 0.0'f32
        for j in 0 ..< scores.len:
          let vRow = j * v.shape[1] + kvh * headDim
          acc += scores[j] * v.data[vRow + d]
        ctx.data[outRow + d] = acc
  matmul(ctx, wo)

proc attentionCached(q: Tensor, kCache, vCache: Tensor, curLen, nHead, nHeadKv, headDim: int): Tensor =
  ## q is [1, nHead*headDim]; caches are [maxLen, nHeadKv*headDim]
  var ctx = newTensor(@[1, nHead * headDim])
  let group = nHead div nHeadKv
  for h in 0 ..< nHead:
    let kvh = h div group
    let hOff = h * headDim
    var scores = newSeq[float32](curLen)
    for j in 0 ..< curLen:
      var dot = 0.0'f32
      let qRow = hOff
      let kRow = j * kCache.shape[1] + kvh * headDim
      for d in 0 ..< headDim:
        dot += q.data[qRow + d] * kCache.data[kRow + d]
      scores[j] = dot / sqrt(float32(headDim))
    var maxv = scores[0]
    for j in 1 ..< scores.len:
      if scores[j] > maxv: maxv = scores[j]
    var sum = 0.0'f32
    for j in 0 ..< scores.len:
      scores[j] = exp(scores[j] - maxv)
      sum += scores[j]
    let inv = 1.0'f32 / sum
    for j in 0 ..< scores.len:
      scores[j] *= inv
    let outRow = hOff
    for d in 0 ..< headDim:
      var acc = 0.0'f32
      for j in 0 ..< scores.len:
        let vRow = j * vCache.shape[1] + kvh * headDim
        acc += scores[j] * vCache.data[vRow + d]
      ctx.data[outRow + d] = acc
  ctx

proc ffn(x: Tensor, wGate, wUp, wDown: Tensor): Tensor =
  let gate = matmul(x, wGate)
  let up = matmul(x, wUp)
  let act = mul(silu(gate), up)
  matmul(act, wDown)

proc linearOut(x: Tensor, w: Tensor, nEmb, nVocab: int): Tensor =
  if w.shape.len != 2:
    raise newException(ValueError, "output weight must be 2D")
  let a0 = w.shape[0]
  let a1 = w.shape[1]
  if a0 == nEmb and a1 == nVocab:
    return matmul(x, w)
  elif a0 == nVocab and a1 == nEmb:
    # treat as rows = vocab
    var outTensor = newTensor(@[x.shape[0], nVocab])
    for i in 0 ..< x.shape[0]:
      let xRow = i * nEmb
      let outRow = i * nVocab
      for v in 0 ..< nVocab:
        var acc = 0.0'f32
        let wRow = v * nEmb
        for d in 0 ..< nEmb:
          acc += x.data[xRow + d] * w.data[wRow + d]
        outTensor.data[outRow + v] = acc
    return outTensor
  else:
    raise newException(ValueError, "output weight shape mismatch")

proc storeKVRange(cache: var KvCache, layer: int, startPos: int, k, v: Tensor) =
  let rows = k.shape[0]
  let rowSize = cache.k[layer].shape[1]
  for r in 0 ..< rows:
    let src = r * rowSize
    let dst = (startPos + r) * rowSize
    for i in 0 ..< rowSize:
      cache.k[layer].data[dst + i] = k.data[src + i]
      cache.v[layer].data[dst + i] = v.data[src + i]

proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
  let hp = m.hparams
  if hp.arch != "" and hp.arch != "llama":
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
    raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
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

    let xNorm = rmsnorm(x, attnNorm, hp.rmsEps)
    var q = matmul(xNorm, wq)
    var k = matmul(xNorm, wk)
    let v = matmul(xNorm, wv)
    applyRopeSingle(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase)
    applyRopeSingle(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase)
    storeKVRange(cache, layer, 0, k, v)
    let attnOut = attentionFull(q, k, v, wo, hp.nHead, hp.nHeadKv, headDim)
    x = add(x, attnOut)

    let xNorm2 = rmsnorm(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffn(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnorm(x, norm, hp.rmsEps)
  result = linearOut(xNormFinal, outW, hp.nEmb, hp.nVocab)

proc forwardDecode*(m: var Model, token: int32, cache: var KvCache): Tensor =
  let hp = m.hparams
  if hp.arch != "" and hp.arch != "llama":
    raise newException(ValueError, "unsupported architecture: " & hp.arch)
  if cache.curLen >= cache.maxLen:
    raise newException(ValueError, "KV cache full")
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

    let xNorm = rmsnorm(x, attnNorm, hp.rmsEps)
    var q = matmul(xNorm, wq)
    var k = matmul(xNorm, wk)
    let v = matmul(xNorm, wv)
    applyRopeAtPos(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos)
    applyRopeAtPos(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos)
    storeKVRange(cache, layer, pos, k, v)
    let attnCtx = attentionCached(q, cache.k[layer], cache.v[layer], pos + 1, hp.nHead, hp.nHeadKv, headDim)
    let attnOut = matmul(attnCtx, wo)
    x = add(x, attnOut)

    let xNorm2 = rmsnorm(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffn(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  cache.curLen = pos + 1
  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnorm(x, norm, hp.rmsEps)
  result = linearOut(xNormFinal, outW, hp.nEmb, hp.nVocab)
