## Minimal LLaMA-style forward pass (float32, CPU, no KV cache).

import
  std/[math],
  ./[model, tensor]

when defined(profileHippo):
  import std/[times, strutils]
  import hippo

  type
    KernelCategory* = enum
      KcEmbedding, KcRmsNormAttn, KcLinearQkv, KcRope, KcKvStore,
      KcAttention, KcLinearO, KcResidualAttn, KcRmsNormFfn,
      KcLinearGateUp, KcSiluMul, KcLinearDown, KcResidualFfn,
      KcFinalNormOutput

    EventPair* = object
      start*: HippoEvent
      stop*: HippoEvent
      cat*: KernelCategory

  proc recordStart(pairs: var seq[EventPair], cat: KernelCategory, stream: HippoStream) =
    var ep: EventPair
    ep.cat = cat
    ep.start = hippoEventCreate()
    hippoEventRecord(ep.start, stream)
    pairs.add(ep)

  proc recordStop(pairs: var seq[EventPair], stream: HippoStream) =
    pairs[^1].stop = hippoEventCreate()
    hippoEventRecord(pairs[^1].stop, stream)

  proc printBreakdown(pairs: seq[EventPair]) =
    var catMs: array[KernelCategory, float32]
    var totalMs: float32 = 0
    for ep in pairs:
      let ms = hippoEventElapsedTime(ep.start, ep.stop)
      catMs[ep.cat] += ms
      totalMs += ms
      hippoEventDestroy(ep.start)
      hippoEventDestroy(ep.stop)
    echo "  GPU kernel breakdown (total=", totalMs.formatFloat(ffDecimal, 2), "ms):"
    for cat in KernelCategory:
      if catMs[cat] > 0:
        let pct = if totalMs > 0: catMs[cat] / totalMs * 100 else: 0.0
        echo "    ", ($cat).alignLeft(20), catMs[cat].formatFloat(ffDecimal, 3), " ms  ",
             pct.formatFloat(ffDecimal, 1), "%"

when defined(useHippo) and defined(useMalebolgia):
  {.error: "useHippo and useMalebolgia are mutually exclusive. Choose one backend.".}

when defined(useMalebolgia):
  import malebolgia

when defined(useHippo):
  import hippo
  import ./forward_hippo

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
  ## Multiply x and w using ggml column layout.
  if x.shape.len != 2 or w.shape.len != 2:
    raise newException(ValueError, "linear: expects 2D tensors")
  let inDim = x.shape[0]
  let seqLen = x.shape[1]
  let wCols = w.shape[0]   # in
  let wRows = w.shape[1]   # out
  if wCols != inDim:
    raise newException(ValueError, "linear: input dim mismatch")
  when defined(useHippo):
    return linearHippoCol(x, w, wCols, wRows, seqLen)
  elif defined(useMalebolgia):
    result = newTensor(@[wRows, seqLen])
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
    result = newTensor(@[wRows, seqLen])
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
    when defined(useHippo):
      gpuCache*: GpuKvCache

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
  when defined(useHippo):
    result.gpuCache = initGpuKvCache(hp.nLayer, hp.nHeadKv, result.headDim, maxLen)

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

when defined(useHippo):
  proc outputWeightForLinear(w: Tensor, nEmb, nVocab: int): Tensor =
    if w.shape.len != 2:
      raise newException(ValueError, "output weight must be 2D")
    let a0 = w.shape[0]
    let a1 = w.shape[1]
    if a0 == nEmb and a1 == nVocab:
      return w
    if a0 == nVocab and a1 == nEmb:
      return w.reshape(@[a1, a0])
    raise newException(ValueError, "output weight shape mismatch")

  proc forwardPrefillHippo(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
    let hp = m.hparams
    if hp.arch != "" and hp.arch != "llama":
      raise newException(ValueError, "unsupported architecture: " & hp.arch)
    if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
      raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
    if tokens.len == 0:
      raise newException(ValueError, "prefill requires at least one token")
    if tokens.len > cache.maxLen:
      raise newException(ValueError, "prefill exceeds KV cache capacity")

    let headDim = hp.nEmb div hp.nHead
    let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
    let kvDim = hp.nHeadKv * headDim
    let seqLen = tokens.len

    ensureGpuContext()
    let maxRows = max(max(hp.nEmb, hp.nFfn), hp.nVocab)
    ensureActivationBuffers(maxRows * seqLen)
    ensureScratchBuffers(maxRows * seqLen)
    let stream = gpuCtx.stream

    let tokEmb = getTensorOr(m, "tok_embeddings.weight", "token_embd.weight")
    let dTokEmb = cachedWeight("token_embd_or_tok_embeddings", tokEmb)
    let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tokens[0], seqLen, stream)

    var xPtr = gpuCtx.act0.devicePtr
    let xNormPtr = gpuCtx.act1.devicePtr
    let tmp0 = gpuCtx.scratch0.devicePtr
    let tmp1 = gpuCtx.scratch1.devicePtr
    let tmp2 = gpuCtx.scratch2.devicePtr

    gpuEmbedding(xPtr, dTokEmb.devicePtr, cast[ptr int32](tokenPtr),
                 hp.nEmb, seqLen, hp.nVocab, stream)

    for layer in 0 ..< hp.nLayer:
      let lp = "blk." & $layer & "."
      let dAttnNorm = cachedWeight(lp & "attn_norm.weight", m.getTensor(lp & "attn_norm.weight"))
      let dFfnNorm = cachedWeight(lp & "ffn_norm.weight", m.getTensor(lp & "ffn_norm.weight"))
      let dWq = cachedWeight(lp & "attn_q.weight", m.getTensor(lp & "attn_q.weight"))
      let dWk = cachedWeight(lp & "attn_k.weight", m.getTensor(lp & "attn_k.weight"))
      let dWv = cachedWeight(lp & "attn_v.weight", m.getTensor(lp & "attn_v.weight"))
      let dWo = cachedWeight(lp & "attn_output.weight", m.getTensor(lp & "attn_output.weight"))
      let dWGate = cachedWeight(lp & "ffn_gate.weight", m.getTensor(lp & "ffn_gate.weight"))
      let dWUp = cachedWeight(lp & "ffn_up.weight", m.getTensor(lp & "ffn_up.weight"))
      let dWDown = cachedWeight(lp & "ffn_down.weight", m.getTensor(lp & "ffn_down.weight"))

      gpuRmsnormCols(xNormPtr, xPtr, dAttnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWq.devicePtr, hp.nEmb, hp.nEmb, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWk.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      gpuLinearCol(tmp2, xNormPtr, dWv.devicePtr, hp.nEmb, kvDim, seqLen, stream)
      gpuRopeAtPos(tmp0, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)
      gpuRopeAtPos(tmp1, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0, seqLen, stream)

      gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)
      gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, kvDim, seqLen, cache.gpuCache.maxLen, 0, stream)

      gpuAttentionPrefill(xNormPtr, tmp0, tmp1, tmp2,
                          hp.nHead, hp.nHeadKv, headDim, seqLen, stream)
      gpuLinearCol(tmp0, xNormPtr, dWo.devicePtr, hp.nEmb, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

      gpuRmsnormCols(xNormPtr, xPtr, dFfnNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
      gpuLinearCol(tmp0, xNormPtr, dWGate.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
      gpuLinearCol(tmp1, xNormPtr, dWUp.devicePtr, hp.nEmb, hp.nFfn, seqLen, stream)
      gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn * seqLen, stream)
      gpuLinearCol(tmp0, tmp2, dWDown.devicePtr, hp.nFfn, hp.nEmb, seqLen, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb * seqLen, stream)

    let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
    let outW = outputWeightForLinear(m.getTensor("output.weight"), hp.nEmb, hp.nVocab)
    let dNorm = cachedWeight("norm_or_output_norm.weight", norm)
    let dOutW = cachedWeight("output.weight", outW)

    gpuRmsnormCols(xNormPtr, xPtr, dNorm.devicePtr, hp.nEmb, seqLen, hp.rmsEps, stream)
    gpuLinearCol(xPtr, xNormPtr, dOutW.devicePtr, outW.shape[0], outW.shape[1], seqLen, stream)

    result = newTensor(@[outW.shape[1], seqLen])
    let bytes = result.data.len * sizeof(float32)
    gpuDownloadFromDevice(addr result.data[0], xPtr, bytes, stream)
    gpuStreamSync(stream)

    cache.curLen = seqLen
    cache.gpuCache.curLen = seqLen

  proc forwardDecodeHippo(m: var Model, token: int32, cache: var KvCache): Tensor =
    let hp = m.hparams
    if hp.arch != "" and hp.arch != "llama":
      raise newException(ValueError, "unsupported architecture: " & hp.arch)
    if hp.nHeadKv != 0 and (hp.nHead mod hp.nHeadKv) != 0:
      raise newException(ValueError, "GQA requires head_count divisible by head_count_kv")
    if cache.curLen >= cache.maxLen:
      raise newException(ValueError, "KV cache full")

    let headDim = hp.nEmb div hp.nHead
    let ropeDim = if hp.ropeDim > 0: hp.ropeDim else: headDim
    let kvDim = hp.nHeadKv * headDim
    let pos = cache.curLen

    ensureGpuContext()
    let maxRows = max(max(hp.nEmb, hp.nFfn), hp.nVocab)
    ensureActivationBuffers(maxRows)
    ensureScratchBuffers(maxRows)
    let stream = gpuCtx.stream

    ensureModelGpuPtrs(m, hp)

    when defined(profileHippo):
      let wallStart = epochTime()
      let gpuStartEvt = hippoEventCreate()
      let gpuEndEvt = hippoEventCreate()
      hippoEventRecord(gpuStartEvt, stream)
      var eventPairs: seq[EventPair]

    var tok = token
    let tokenPtr = gpuUploadInt32Pooled(unsafeAddr tok, 1, stream)

    var xPtr = gpuCtx.act0.devicePtr
    let xNormPtr = gpuCtx.act1.devicePtr
    let tmp0 = gpuCtx.scratch0.devicePtr
    let tmp1 = gpuCtx.scratch1.devicePtr
    let tmp2 = gpuCtx.scratch2.devicePtr

    when defined(profileHippo):
      recordStart(eventPairs, KcEmbedding, stream)
    gpuEmbedding(xPtr, modelPtrs.tokEmb, cast[ptr int32](tokenPtr),
                 hp.nEmb, 1, hp.nVocab, stream)
    when defined(profileHippo):
      recordStop(eventPairs, stream)

    when defined(profileHippo):
      var kernelLaunchMs = 0.0

    for layer in 0 ..< hp.nLayer:
      let lw = modelPtrs.layers[layer]

      when defined(profileHippo):
        let klStart = epochTime()
        recordStart(eventPairs, KcRmsNormAttn, stream)
      gpuRmsnormCols(xNormPtr, xPtr, lw.attnNorm, hp.nEmb, 1, hp.rmsEps, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearQkv, stream)
      # Q/K/V projections — dispatch quantized or float32
      if lw.wqQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, hp.nEmb, lw.wqQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, hp.nEmb, 1, stream)
      if lw.wkQ != nil:
        gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
      else:
        gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
      if lw.wvQ != nil:
        gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
      else:
        gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcRope, stream)
      gpuRopeAtPos(tmp0, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos, 1, stream)
      gpuRopeAtPos(tmp1, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, 1, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcKvStore, stream)
      gpuStoreKV(cache.gpuCache.k[layer].devicePtr, tmp1, kvDim, 1, cache.gpuCache.maxLen, pos, stream)
      gpuStoreKV(cache.gpuCache.v[layer].devicePtr, tmp2, kvDim, 1, cache.gpuCache.maxLen, pos, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcAttention, stream)
      gpuAttentionDecode(xNormPtr, tmp0, cache.gpuCache.k[layer].devicePtr,
                         cache.gpuCache.v[layer].devicePtr,
                         hp.nHead, hp.nHeadKv, headDim, pos + 1,
                         cache.gpuCache.maxLen, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearO, stream)
      if lw.woQ != nil:
        gpuLinearColQuant(tmp0, xNormPtr, lw.woQ, hp.nEmb, hp.nEmb, lw.woQType, stream)
      else:
        gpuLinearCol(tmp0, xNormPtr, lw.wo, hp.nEmb, hp.nEmb, 1, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcResidualAttn, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcRmsNormFfn, stream)
      gpuRmsnormCols(xNormPtr, xPtr, lw.ffnNorm, hp.nEmb, 1, hp.rmsEps, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearGateUp, stream)
      when HippoWarpSize == 32:
        if lw.wGateQType == GgmlTypeQ3K and lw.wUpQType == GgmlTypeQ3K:
          # Fused gate+up+silu: single kernel launch, no intermediate buffers
          gpuFusedGateUpSiluQ3K(tmp2, xNormPtr, lw.wGateQ, lw.wUpQ,
                                hp.nEmb, hp.nFfn, stream)
        else:
          if lw.wGateQ != nil:
            gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
          else:
            gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
          if lw.wUpQ != nil:
            gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
          else:
            gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
          when defined(profileHippo):
            recordStop(eventPairs, stream)
            recordStart(eventPairs, KcSiluMul, stream)
          gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      else:
        if lw.wGateQ != nil:
          gpuLinearColQuant(tmp0, xNormPtr, lw.wGateQ, hp.nEmb, hp.nFfn, lw.wGateQType, stream)
        else:
          gpuLinearCol(tmp0, xNormPtr, lw.wGate, hp.nEmb, hp.nFfn, 1, stream)
        if lw.wUpQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wUpQ, hp.nEmb, hp.nFfn, lw.wUpQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wUp, hp.nEmb, hp.nFfn, 1, stream)
        when defined(profileHippo):
          recordStop(eventPairs, stream)
          recordStart(eventPairs, KcSiluMul, stream)
        gpuSiluMul(tmp2, tmp0, tmp1, hp.nFfn, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcLinearDown, stream)
      if lw.wDownQ != nil:
        gpuLinearColQuant(tmp0, tmp2, lw.wDownQ, hp.nFfn, hp.nEmb, lw.wDownQType, stream)
      else:
        gpuLinearCol(tmp0, tmp2, lw.wDown, hp.nFfn, hp.nEmb, 1, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        recordStart(eventPairs, KcResidualFfn, stream)
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb, stream)
      when defined(profileHippo):
        recordStop(eventPairs, stream)
        kernelLaunchMs += (epochTime() - klStart) * 1000

    when defined(profileHippo):
      recordStart(eventPairs, KcFinalNormOutput, stream)
    gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.normWeight, hp.nEmb, 1, hp.rmsEps, stream)
    gpuLinearCol(xPtr, xNormPtr, modelPtrs.outputWeight, modelPtrs.outputShape0, modelPtrs.outputShape1, 1, stream)
    when defined(profileHippo):
      recordStop(eventPairs, stream)

    when defined(profileHippo):
      hippoEventRecord(gpuEndEvt, stream)
      let preSyncWall = epochTime()

    result = newTensor(@[modelPtrs.outputShape1, 1])
    let bytes = result.data.len * sizeof(float32)
    gpuDownloadFromDevice(addr result.data[0], xPtr, bytes, stream)
    gpuStreamSync(stream)

    when defined(profileHippo):
      let wallEnd = epochTime()
      hippoEventSynchronize(gpuEndEvt)
      printBreakdown(eventPairs)
      let gpuMs = hippoEventElapsedTime(gpuStartEvt, gpuEndEvt)
      let totalMs = (wallEnd - wallStart) * 1000
      let cpuMs = (preSyncWall - wallStart) * 1000
      let syncMs = (wallEnd - preSyncWall) * 1000
      echo "decode[pos=", pos, "]: total=", totalMs.formatFloat(ffDecimal, 1), "ms",
           " cpu=", cpuMs.formatFloat(ffDecimal, 1), "ms",
           " sync=", syncMs.formatFloat(ffDecimal, 1), "ms",
           " gpu=", gpuMs.formatFloat(ffDecimal, 1), "ms",
           " kernelLaunch=", kernelLaunchMs.formatFloat(ffDecimal, 1), "ms"
      hippoEventDestroy(gpuStartEvt)
      hippoEventDestroy(gpuEndEvt)

    cache.curLen = pos + 1
    cache.gpuCache.curLen = cache.curLen

proc forwardPrefill*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
  when defined(useHippo):
    return forwardPrefillHippo(m, tokens, cache)
  else:
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
  when defined(useHippo):
    return forwardDecodeHippo(m, token, cache)
  else:
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
