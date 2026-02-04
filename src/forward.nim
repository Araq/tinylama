## Minimal LLaMA-style forward pass (float32, CPU, no KV cache).

import std/[math]
import ./tensor
import ./model
when defined(useMalebolgia):
  import malebolgia

proc getTensorOr(m: var Model, a, b: string): Tensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

proc embeddingLookup(weight: Tensor, tokens: seq[int32], nVocab, nEmb: int): Tensor =
  ## Returns [nEmb, seq] in ggml-style column layout.
  result = newTensor(@[nEmb, tokens.len])
  if weight.shape.len != 2:
    raise newException(ValueError, "embedding weight must be 2D")
  let a0 = weight.shape[0]
  let a1 = weight.shape[1]
  if a0 == nVocab and a1 == nEmb:
    let rowSize = a1
    for i, t in tokens:
      let tid = int(t)
      let row = tid * rowSize
      for e in 0 ..< nEmb:
        result.data[e * tokens.len + i] = weight.data[row + e]
  elif a0 == nEmb and a1 == nVocab:
    # ggml layout: rows = a1 (vocab), cols = a0 (emb)
    let rowSize = a0
    for i, t in tokens:
      let tid = int(t)
      let row = tid * rowSize
      for e in 0 ..< nEmb:
        result.data[e * tokens.len + i] = weight.data[row + e]
  else:
    raise newException(ValueError, "embedding shape mismatch")

proc linearGGMLCol(x: Tensor, w: Tensor): Tensor =
  ## x: [in, seq], w stored as [in, out] but laid out row-major as rows=out, cols=in.
  if x.shape.len != 2 or w.shape.len != 2:
    raise newException(ValueError, "linear: expects 2D tensors")
  let inDim = x.shape[0]
  let seqLen = x.shape[1]
  let wCols = w.shape[0]   # in
  let wRows = w.shape[1]   # out
  if wCols != inDim:
    raise newException(ValueError, "linear: input dim mismatch")
  result = newTensor(@[wRows, seqLen])
  when defined(useMalebolgia):
    proc linearChunk(startRow, endRow: int,
                     wData, xData, outData: ptr UncheckedArray[float32],
                     wCols, seqLen: int) {.gcsafe.} =
      for o in startRow ..< endRow:
        let wRow = o * wCols
        let outRow = o * seqLen
        for s in 0 ..< seqLen:
          var acc = 0.0'f32
          for k in 0 ..< wCols:
            acc += wData[wRow + k] * xData[k * seqLen + s]
          outData[outRow + s] = acc

    let wData = cast[ptr UncheckedArray[float32]](addr w.data[0])
    let xData = cast[ptr UncheckedArray[float32]](addr x.data[0])
    let outData = cast[ptr UncheckedArray[float32]](addr result.data[0])
    let chunk = if wRows >= 64: 32 else: wRows
    var m = createMaster()
    m.awaitAll:
      var start = 0
      while start < wRows:
        let stop = min(start + chunk, wRows)
        m.spawn linearChunk(start, stop, wData, xData, outData, wCols, seqLen)
        start = stop
  else:
    for o in 0 ..< wRows:
      let wRow = o * wCols
      let outRow = o * seqLen
      for s in 0 ..< seqLen:
        var acc = 0.0'f32
        for k in 0 ..< inDim:
          acc += w.data[wRow + k] * x.data[k * seqLen + s]
        result.data[outRow + s] = acc

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
    result.k[i] = newTensor(@[kvDim, maxLen])
    result.v[i] = newTensor(@[kvDim, maxLen])

proc applyRopeSingle(x: var Tensor, nHead, headDim, ropeDim: int, base: float32) =
  let seqLen = x.shape[1]
  for h in 0 ..< nHead:
    let hOffset = h * headDim
    for p in 0 ..< seqLen:
      for i in 0 ..< ropeDim div 2:
        let idx0 = (hOffset + 2 * i) * seqLen + p
        let idx1 = (hOffset + 2 * i + 1) * seqLen + p
        let theta = pow(1.0'f32 / base, float32(2 * i) / float32(ropeDim))
        let angle = float32(p) * theta
        let c = cos(angle)
        let s = sin(angle)
        let v0 = x.data[idx0]
        let v1 = x.data[idx1]
        x.data[idx0] = v0 * c - v1 * s
        x.data[idx1] = v0 * s + v1 * c

proc applyRopeAtPos(x: var Tensor, nHead, headDim, ropeDim: int, base: float32, pos: int) =
  let seqLen = x.shape[1]
  for h in 0 ..< nHead:
    let hOffset = h * headDim
    for i in 0 ..< ropeDim div 2:
      let idx0 = (hOffset + 2 * i) * seqLen
      let idx1 = (hOffset + 2 * i + 1) * seqLen
      let theta = pow(1.0'f32 / base, float32(2 * i) / float32(ropeDim))
      let angle = float32(pos) * theta
      let c = cos(angle)
      let s = sin(angle)
      let v0 = x.data[idx0]
      let v1 = x.data[idx1]
      x.data[idx0] = v0 * c - v1 * s
      x.data[idx1] = v0 * s + v1 * c

proc attentionFull(q, k, v, wo: Tensor, nHead, nHeadKv, headDim: int): Tensor =
  let seqLen = q.shape[1]

  var ctx = newTensor(@[nHead * headDim, seqLen])
  let group = nHead div nHeadKv
  for h in 0 ..< nHead:
    let kvh = h div group
    let hOff = h * headDim
    for i in 0 ..< seqLen:
      var scores = newSeq[float32](i + 1)
      for j in 0 ..< i + 1:
        var dot = 0.0'f32
        for d in 0 ..< headDim:
          let qIdx = (hOff + d) * seqLen + i
          let kIdx = (kvh * headDim + d) * seqLen + j
          dot += q.data[qIdx] * k.data[kIdx]
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
      for d in 0 ..< headDim:
        var acc = 0.0'f32
        for j in 0 ..< scores.len:
          let vIdx = (kvh * headDim + d) * seqLen + j
          acc += scores[j] * v.data[vIdx]
        let outIdx = (hOff + d) * seqLen + i
        ctx.data[outIdx] = acc
  linearGGMLCol(ctx, wo)

proc attentionCached(q: Tensor, kCache, vCache: Tensor, curLen, nHead, nHeadKv, headDim: int): Tensor =
  ## q is [nHead*headDim, 1]; caches are [kvDim, maxLen]
  var ctx = newTensor(@[nHead * headDim, 1])
  let group = nHead div nHeadKv
  for h in 0 ..< nHead:
    let kvh = h div group
    let hOff = h * headDim
    var scores = newSeq[float32](curLen)
    for j in 0 ..< curLen:
      var dot = 0.0'f32
      for d in 0 ..< headDim:
        let qIdx = (hOff + d)
        let kIdx = (kvh * headDim + d) * kCache.shape[1] + j
        dot += q.data[qIdx] * kCache.data[kIdx]
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
    for d in 0 ..< headDim:
      var acc = 0.0'f32
      for j in 0 ..< scores.len:
        let vIdx = (kvh * headDim + d) * vCache.shape[1] + j
        acc += scores[j] * vCache.data[vIdx]
      let outIdx = (hOff + d)
      ctx.data[outIdx] = acc
  ctx

proc ffn(x: Tensor, wGate, wUp, wDown: Tensor): Tensor =
  let gate = linearGGMLCol(x, wGate)
  let up = linearGGMLCol(x, wUp)
  let act = mul(silu(gate), up)
  linearGGMLCol(act, wDown)

proc linearOut(x: Tensor, w: Tensor, nEmb, nVocab: int): Tensor =
  if w.shape.len != 2:
    raise newException(ValueError, "output weight must be 2D")
  let a0 = w.shape[0]
  let a1 = w.shape[1]
  if a0 == nEmb and a1 == nVocab:
    return linearGGMLCol(x, w)
  elif a0 == nVocab and a1 == nEmb:
    # treat as rows = vocab
    return linearGGMLCol(x, w.reshape(@[a1, a0]))
  else:
    raise newException(ValueError, "output weight shape mismatch")

proc storeKVRange(cache: var KvCache, layer: int, startPos: int, k, v: Tensor) =
  let rows = k.shape[0]
  let cols = k.shape[1]
  let cacheCols = cache.k[layer].shape[1]
  for r in 0 ..< rows:
    let src = r * cols
    let dst = r * cacheCols + startPos
    for c in 0 ..< cols:
      cache.k[layer].data[dst + c] = k.data[src + c]
      cache.v[layer].data[dst + c] = v.data[src + c]

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

    let xNorm = rmsnormCols(x, attnNorm, hp.rmsEps)
    var q = linearGGMLCol(xNorm, wq)
    var k = linearGGMLCol(xNorm, wk)
    let v = linearGGMLCol(xNorm, wv)
    applyRopeSingle(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase)
    applyRopeSingle(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase)
    storeKVRange(cache, layer, 0, k, v)
    let attnOut = attentionFull(q, k, v, wo, hp.nHead, hp.nHeadKv, headDim)
    x = add(x, attnOut)

    let xNorm2 = rmsnormCols(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffn(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, norm, hp.rmsEps)
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

    let xNorm = rmsnormCols(x, attnNorm, hp.rmsEps)
    var q = linearGGMLCol(xNorm, wq)
    var k = linearGGMLCol(xNorm, wk)
    let v = linearGGMLCol(xNorm, wv)
    applyRopeAtPos(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos)
    applyRopeAtPos(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos)
    storeKVRange(cache, layer, pos, k, v)
    let attnCtx = attentionCached(q, cache.k[layer], cache.v[layer], pos + 1, hp.nHead, hp.nHeadKv, headDim)
    let attnOut = linearGGMLCol(attnCtx, wo)
    x = add(x, attnOut)

    let xNorm2 = rmsnormCols(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffn(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  cache.curLen = pos + 1
  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, norm, hp.rmsEps)
  result = linearOut(xNormFinal, outW, hp.nEmb, hp.nVocab)
