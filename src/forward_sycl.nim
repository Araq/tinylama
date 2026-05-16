## SYCL backend scaffold with relayable bulk operations.
## This first stage keeps CPU math implementations behind SYCL relays so
## we can swap in oneAPI kernels operation-by-operation without touching
## frontend inference flow.

import std/[math, tables]
import ./[forward_types, model, tensor]

when not defined(cpp):
  {.error: "useSycl requires Nim's C++ backend. Build with `nim cpp`.".}
when defined(useSycl):
  {.passC: "-DTINYLAMA_REQUIRE_SYCL -DTINYLAMA_ENABLE_SYCL".}
  {.compile: "cpp/sycl_linear.cpp".}
when defined(useSyclOneapi):
  {.warning: "useSyclOneapi selected: pass oneAPI flags/toolchain explicitly (e.g. icpx + -fsycl).".}

type
  SyclLinearRelay* = proc(x: Tensor, w: Tensor): Tensor {.nimcall.}
  SyclRmsNormRelay* = proc(x: Tensor, w: Tensor, eps: float32): Tensor {.nimcall.}
  SyclRopeSingleRelay* = proc(x: var Tensor, nHead, headDim, ropeDim: int, base: float32) {.nimcall.}
  SyclRopeAtPosRelay* = proc(x: var Tensor, nHead, headDim, ropeDim: int, base: float32, pos: int) {.nimcall.}
  SyclStoreKvRelay* = proc(cache: var KvCache, layer: int, startPos: int, k, v: Tensor) {.nimcall.}
  SyclAttentionFullRelay* = proc(q, k, v, wo: Tensor, nHead, nHeadKv, headDim: int): Tensor {.nimcall.}
  SyclAttentionCachedRelay* = proc(q: Tensor, kCache, vCache: Tensor, curLen, nHead, nHeadKv, headDim: int): Tensor {.nimcall.}

var
  syclLinearRelay*: SyclLinearRelay
  syclRmsNormRelay*: SyclRmsNormRelay
  syclRopeSingleRelay*: SyclRopeSingleRelay
  syclRopeAtPosRelay*: SyclRopeAtPosRelay
  syclStoreKvRelay*: SyclStoreKvRelay
  syclAttentionFullRelay*: SyclAttentionFullRelay
  syclAttentionCachedRelay*: SyclAttentionCachedRelay
  syclOpsInitialized = false

proc syclBackendAvailableRaw(): cint {.importc: "tinylama_sycl_backend_available", cdecl.}
proc syclLinearF32Raw(xPtr, wPtr, outPtr: ptr float32, inDim, seqLen, outRows: cint): cint
  {.importc: "tinylama_sycl_linear_f32", cdecl.}
proc syclRmsNormColsF32Raw(xPtr, weightPtr, outPtr: ptr float32, dim, seqLen: cint, eps: cfloat): cint
  {.importc: "tinylama_sycl_rmsnorm_cols_f32", cdecl.}
proc syclStoreKvColsF32Raw(cachePtr, srcPtr: ptr float32, rows, srcCols, cacheCols, startPos: cint): cint
  {.importc: "tinylama_sycl_store_kv_cols_f32", cdecl.}
proc syclAttentionDecodeF32Raw(qPtr, kCachePtr, vCachePtr, outPtr: ptr float32,
                               nHead, nHeadKv, headDim, curLen, cacheCols: cint): cint
  {.importc: "tinylama_sycl_attention_decode_f32", cdecl.}
proc syclRopeAtPosF32Raw(xPtr: ptr float32, nHead, headDim, ropeDim: cint,
                         base: cfloat, pos, seqLen: cint): cint
  {.importc: "tinylama_sycl_rope_at_pos_f32", cdecl.}
proc syclAttentionPrefillF32Raw(qPtr, kPtr, vPtr, outPtr: ptr float32,
                                nHead, nHeadKv, headDim, seqLen: cint): cint
  {.importc: "tinylama_sycl_attention_prefill_f32", cdecl.}

proc isSupportedArch(arch: string): bool =
  arch.len == 0 or arch == "llama" or arch == "qwen2"

proc getTensorOr(m: var Model, a, b: string): Tensor =
  try:
    return m.getTensor(a)
  except KeyError:
    return m.getTensor(b)

proc getTensorAny(m: var Model, names: openArray[string]): Tensor =
  for n in names:
    if m.infos.hasKey(n):
      return m.getTensor(n)
  raise newException(KeyError, "missing tensor: none of the candidate names exist")

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
    let rowSize = a0
    for i, t in tokens:
      let tid = int(t)
      let row = tid * rowSize
      for e in 0 ..< nEmb:
        result.data[e * tokens.len + i] = weight.data[row + e]
  else:
    raise newException(ValueError, "embedding shape mismatch")

proc linearSycl(x: Tensor, w: Tensor): Tensor =
  ## Uses SYCL kernel and fails hard if unavailable.
  if x.shape.len != 2 or w.shape.len != 2:
    raise newException(ValueError, "linear: expects 2D tensors")
  let inDim = x.shape[0]
  let seqLen = x.shape[1]
  let wCols = w.shape[0]
  let wRows = w.shape[1]
  if wCols != inDim:
    raise newException(ValueError, "linear: input dim mismatch")
  result = newTensor(@[wRows, seqLen])
  if result.data.len == 0:
    return result

  if syclBackendAvailableRaw() != 0:
    let ok = syclLinearF32Raw(
      addr x.data[0],
      addr w.data[0],
      addr result.data[0],
      inDim.cint,
      seqLen.cint,
      wRows.cint
    )
    if ok != 0:
      return result
    raise newException(ValueError, "SYCL backend active but linear kernel execution failed")

  raise newException(ValueError, "SYCL backend is unavailable. Build with a SYCL toolchain (e.g. icpx -fsycl)")

proc rmsNormSycl(x: Tensor, w: Tensor, eps: float32): Tensor =
  if x.shape.len != 2:
    raise newException(ValueError, "rmsnorm: x must be 2D")
  if w.shape.len != 1:
    raise newException(ValueError, "rmsnorm: weight must be 1D")
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  if w.shape[0] != dim:
    raise newException(ValueError, "rmsnorm: weight dim mismatch")
  result = newTensor(@[dim, seqLen])
  if result.data.len == 0:
    return result

  if syclBackendAvailableRaw() != 0:
    let ok = syclRmsNormColsF32Raw(
      addr x.data[0],
      addr w.data[0],
      addr result.data[0],
      dim.cint,
      seqLen.cint,
      eps.cfloat
    )
    if ok != 0:
      return result
    raise newException(ValueError, "SYCL backend active but RMSNorm kernel execution failed")

  raise newException(ValueError, "SYCL backend is unavailable. Build with a SYCL toolchain (e.g. icpx -fsycl)")

proc storeKvSycl(cache: var KvCache, layer: int, startPos: int, k, v: Tensor) =
  let rows = k.shape[0]
  let cols = k.shape[1]
  let cacheCols = cache.k[layer].shape[1]
  if v.shape.len != 2 or v.shape[0] != rows or v.shape[1] != cols:
    raise newException(ValueError, "storeKV: K/V shape mismatch")
  if startPos < 0 or startPos + cols > cacheCols:
    raise newException(ValueError, "storeKV: range out of cache bounds")
  let kOk = syclStoreKvColsF32Raw(
    addr cache.k[layer].data[0],
    addr k.data[0],
    rows.cint,
    cols.cint,
    cacheCols.cint,
    startPos.cint
  )
  if kOk == 0:
    raise newException(ValueError, "SYCL KV-store kernel failed for K cache")
  let vOk = syclStoreKvColsF32Raw(
    addr cache.v[layer].data[0],
    addr v.data[0],
    rows.cint,
    cols.cint,
    cacheCols.cint,
    startPos.cint
  )
  if vOk == 0:
    raise newException(ValueError, "SYCL KV-store kernel failed for V cache")

proc attentionCachedSycl(q: Tensor, kCache, vCache: Tensor, curLen, nHead, nHeadKv, headDim: int): Tensor =
  ## Uses SYCL decode-attention kernel and fails hard if execution fails.
  if q.shape.len != 2 or q.shape[1] != 1:
    raise newException(ValueError, "attention decode: q must be [nHead*headDim, 1]")
  if kCache.shape.len != 2 or vCache.shape.len != 2:
    raise newException(ValueError, "attention decode: cache tensors must be 2D")
  if kCache.shape != vCache.shape:
    raise newException(ValueError, "attention decode: K/V cache shape mismatch")
  result = newTensor(@[nHead * headDim, 1])
  if result.data.len == 0:
    return result
  let ok = syclAttentionDecodeF32Raw(
    addr q.data[0],
    addr kCache.data[0],
    addr vCache.data[0],
    addr result.data[0],
    nHead.cint,
    nHeadKv.cint,
    headDim.cint,
    curLen.cint,
    kCache.shape[1].cint
  )
  if ok == 0:
    raise newException(ValueError, "SYCL attention decode kernel failed")

proc ropeSingleSycl(x: var Tensor, nHead, headDim, ropeDim: int, base: float32) =
  if x.shape.len != 2:
    raise newException(ValueError, "RoPE: tensor must be 2D")
  let seqLen = x.shape[1]
  let ok = syclRopeAtPosF32Raw(
    addr x.data[0],
    nHead.cint,
    headDim.cint,
    ropeDim.cint,
    base.cfloat,
    0.cint,
    seqLen.cint
  )
  if ok == 0:
    raise newException(ValueError, "SYCL RoPE kernel failed (prefill path)")

proc ropeAtPosSycl(x: var Tensor, nHead, headDim, ropeDim: int, base: float32, pos: int) =
  if x.shape.len != 2:
    raise newException(ValueError, "RoPE: tensor must be 2D")
  let seqLen = x.shape[1]
  let ok = syclRopeAtPosF32Raw(
    addr x.data[0],
    nHead.cint,
    headDim.cint,
    ropeDim.cint,
    base.cfloat,
    pos.cint,
    seqLen.cint
  )
  if ok == 0:
    raise newException(ValueError, "SYCL RoPE kernel failed (decode path)")

proc attentionFullSycl(q, k, v, wo: Tensor, nHead, nHeadKv, headDim: int): Tensor =
  if q.shape.len != 2 or k.shape.len != 2 or v.shape.len != 2:
    raise newException(ValueError, "attention prefill: q/k/v must be 2D")
  let seqLen = q.shape[1]
  let outDim = nHead * headDim
  var ctx = newTensor(@[outDim, seqLen])
  let ok = syclAttentionPrefillF32Raw(
    addr q.data[0],
    addr k.data[0],
    addr v.data[0],
    addr ctx.data[0],
    nHead.cint,
    nHeadKv.cint,
    headDim.cint,
    seqLen.cint
  )
  if ok == 0:
    raise newException(ValueError, "SYCL attention prefill kernel failed")
  syclLinearRelay(ctx, wo)

proc initSyclOps*() =
  if syclBackendAvailableRaw() == 0:
    raise newException(ValueError,
      "useSycl requires a working SYCL toolchain/runtime; no CPU fallback is allowed in this mode")
  syclLinearRelay = linearSycl
  syclRmsNormRelay = rmsNormSycl
  syclRopeSingleRelay = ropeSingleSycl
  syclRopeAtPosRelay = ropeAtPosSycl
  syclStoreKvRelay = storeKvSycl
  syclAttentionFullRelay = attentionFullSycl
  syclAttentionCachedRelay = attentionCachedSycl
  syclOpsInitialized = true

proc ensureSyclOps() =
  if not syclOpsInitialized:
    initSyclOps()

proc ffnSycl(x: Tensor, wGate, wUp, wDown: Tensor): Tensor =
  let gate = syclLinearRelay(x, wGate)
  let up = syclLinearRelay(x, wUp)
  let act = mul(silu(gate), up)
  syclLinearRelay(act, wDown)

proc linearOutSycl(x: Tensor, w: Tensor, nEmb, nVocab: int): Tensor =
  if w.shape.len != 2:
    raise newException(ValueError, "output weight must be 2D")
  let a0 = w.shape[0]
  let a1 = w.shape[1]
  if a0 == nEmb and a1 == nVocab:
    return syclLinearRelay(x, w)
  elif a0 == nVocab and a1 == nEmb:
    return syclLinearRelay(x, w.reshape(@[a1, a0]))
  else:
    raise newException(ValueError, "output weight shape mismatch")

proc forwardPrefillSycl*(m: var Model, tokens: seq[int32], cache: var KvCache): Tensor =
  ensureSyclOps()
  let hp = m.hparams
  if not isSupportedArch(hp.arch):
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

    let xNorm = syclRmsNormRelay(x, attnNorm, hp.rmsEps)
    var q = syclLinearRelay(xNorm, wq)
    var k = syclLinearRelay(xNorm, wk)
    let v = syclLinearRelay(xNorm, wv)
    syclRopeSingleRelay(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase)
    syclRopeSingleRelay(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase)
    syclStoreKvRelay(cache, layer, 0, k, v)
    let attnOut = syclAttentionFullRelay(q, k, v, wo, hp.nHead, hp.nHeadKv, headDim)
    x = add(x, attnOut)

    let xNorm2 = syclRmsNormRelay(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffnSycl(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = getTensorAny(m, ["output.weight", "token_embd.weight", "tok_embeddings.weight"])
  let xNormFinal = syclRmsNormRelay(x, norm, hp.rmsEps)
  result = linearOutSycl(xNormFinal, outW, hp.nEmb, hp.nVocab)

proc forwardDecodeSycl*(m: var Model, token: int32, cache: var KvCache): Tensor =
  ensureSyclOps()
  let hp = m.hparams
  if not isSupportedArch(hp.arch):
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

    let xNorm = syclRmsNormRelay(x, attnNorm, hp.rmsEps)
    var q = syclLinearRelay(xNorm, wq)
    var k = syclLinearRelay(xNorm, wk)
    let v = syclLinearRelay(xNorm, wv)
    syclRopeAtPosRelay(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos)
    syclRopeAtPosRelay(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos)
    syclStoreKvRelay(cache, layer, pos, k, v)
    let attnCtx = syclAttentionCachedRelay(q, cache.k[layer], cache.v[layer], pos + 1, hp.nHead, hp.nHeadKv, headDim)
    let attnOut = syclLinearRelay(attnCtx, wo)
    x = add(x, attnOut)

    let xNorm2 = syclRmsNormRelay(x, ffnNorm, hp.rmsEps)
    let ffnOut = ffnSycl(xNorm2, wGate, wUp, wDown)
    x = add(x, ffnOut)

  cache.curLen = pos + 1
  let norm = getTensorOr(m, "norm.weight", "output_norm.weight")
  let outW = getTensorAny(m, ["output.weight", "token_embd.weight", "tok_embeddings.weight"])
  let xNormFinal = syclRmsNormRelay(x, norm, hp.rmsEps)
  result = linearOutSycl(xNormFinal, outW, hp.nEmb, hp.nVocab)
