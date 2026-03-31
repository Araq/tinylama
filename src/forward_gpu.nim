## GPU forward pass orchestrators (hippo backend).
## This file is only compiled when -d:useHippo is set.

import
  hippo,
  ./[forward_hippo, forward_types, model, tensor]

when defined(profileHippo):
  import std/[times, strutils]

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

proc getTensorOr(m: var Model, a, b: string): Tensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

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

proc forwardPrefillGpu*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
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

proc forwardDecodeGpu*(m: var Model, token: int32, cache: var KvCache): Tensor =
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

  # First layer's attn rmsnorm (not fused with a residual)
  when defined(profileHippo):
    var kernelLaunchMs0 = 0.0
    let klStart0 = epochTime()
    recordStart(eventPairs, KcRmsNormAttn, stream)
  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.layers[0].attnNorm, hp.nEmb, 1, hp.rmsEps, stream)
  when defined(profileHippo):
    recordStop(eventPairs, stream)
    kernelLaunchMs0 += (epochTime() - klStart0) * 1000

  for layer in 0 ..< hp.nLayer:
    let lw = modelPtrs.layers[layer]

    when defined(profileHippo):
      let klStart = epochTime()
      recordStart(eventPairs, KcLinearQkv, stream)
    # Q/K/V projections — dispatch quantized or float32
    if lw.wqQ != nil:
      gpuLinearColQuant(tmp0, xNormPtr, lw.wqQ, hp.nEmb, hp.nEmb, lw.wqQType, stream)
    else:
      gpuLinearCol(tmp0, xNormPtr, lw.wq, hp.nEmb, hp.nEmb, 1, stream)
    when HippoWarpSize == 32:
      if lw.wkQ != nil and lw.wvQ != nil and lw.wkQType == GgmlTypeQ2K and lw.wvQType == GgmlTypeQ3K:
        gpuFusedKVLinearQ2KQ3K(tmp1, tmp2, xNormPtr, lw.wkQ, lw.wvQ, hp.nEmb, kvDim, stream)
      else:
        if lw.wkQ != nil:
          gpuLinearColQuant(tmp1, xNormPtr, lw.wkQ, hp.nEmb, kvDim, lw.wkQType, stream)
        else:
          gpuLinearCol(tmp1, xNormPtr, lw.wk, hp.nEmb, kvDim, 1, stream)
        if lw.wvQ != nil:
          gpuLinearColQuant(tmp2, xNormPtr, lw.wvQ, hp.nEmb, kvDim, lw.wvQType, stream)
        else:
          gpuLinearCol(tmp2, xNormPtr, lw.wv, hp.nEmb, kvDim, 1, stream)
    else:
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
    gpuRopeQKDecode(tmp0, tmp1, hp.nHead, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, stream)
    when defined(profileHippo):
      recordStop(eventPairs, stream)
      recordStart(eventPairs, KcKvStore, stream)
    gpuStoreKVPair(cache.gpuCache.k[layer].devicePtr, tmp1,
                    cache.gpuCache.v[layer].devicePtr, tmp2,
                    kvDim, 1, cache.gpuCache.maxLen, pos, stream)
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
    # Fused: x += tmp0; xNorm = rmsnorm(x, ffnNorm)
    gpuResidualRmsnorm(xNormPtr, xPtr, tmp0, lw.ffnNorm, hp.nEmb, hp.rmsEps, stream)
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
    if layer < hp.nLayer - 1:
      # Fused: x += tmp0; xNorm = rmsnorm(x, nextLayer.attnNorm)
      gpuResidualRmsnorm(xNormPtr, xPtr, tmp0,
                          modelPtrs.layers[layer + 1].attnNorm,
                          hp.nEmb, hp.rmsEps, stream)
    else:
      # Last layer: plain residual, rmsnorm happens below with final norm weight
      gpuAdd(xPtr, xPtr, tmp0, hp.nEmb, stream)
    when defined(profileHippo):
      recordStop(eventPairs, stream)
      kernelLaunchMs += (epochTime() - klStart) * 1000

  when defined(profileHippo):
    recordStart(eventPairs, KcFinalNormOutput, stream)
  gpuRmsnormCols(xNormPtr, xPtr, modelPtrs.normWeight, hp.nEmb, 1, hp.rmsEps, stream)
  if modelPtrs.outputWeightQ != nil:
    gpuLinearColQuant(xPtr, xNormPtr, modelPtrs.outputWeightQ,
                      modelPtrs.outputShape0, modelPtrs.outputShape1,
                      modelPtrs.outputQType, stream)
  else:
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
