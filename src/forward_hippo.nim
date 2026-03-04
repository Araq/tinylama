## Hippo GPU backend: kernels and GPU-resident forward pass.
##
## All model weights, activations, and KV caches live on GPU.
## Only the initial token IDs are uploaded and final logits downloaded.

import
  std/tables,
  ./tensor

when not defined(cpp):
  {.error: "useHippo requires Nim's C++ backend. Build with `nim cpp`.".}

import hippo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
const
  HippoBlockSize = 256
  HippoBlockSizeX = 16
  HippoBlockSizeY = 16

type
  HippoAllocRef* = type(hippoMalloc(1))

# ---------------------------------------------------------------------------
# GpuTensor – a device-resident tensor
# ---------------------------------------------------------------------------
type
  GpuTensor* = object
    devicePtr*: pointer       # raw device pointer
    alloc: HippoAllocRef      # ref-counted allocation (prevents free)
    shape*: seq[int]
    sizeBytes*: int

proc newGpuTensor*(shape: seq[int]): GpuTensor =
  ## Allocate a new GPU tensor with the given shape.
  var total = 1
  for s in shape: total *= s
  let bytes = total * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  result = GpuTensor(
    devicePtr: alloc.p,
    alloc: alloc,
    shape: shape,
    sizeBytes: bytes,
  )

proc numel*(gt: GpuTensor): int =
  result = 1
  for s in gt.shape: result *= s

proc uploadToGpu*(t: Tensor, stream: HippoStream): GpuTensor =
  ## Upload a CPU tensor to a new GPU tensor.
  let bytes = t.data.len * sizeof(float32)
  let alloc = hippoMalloc(bytes)
  hippoMemcpyAsync(alloc.p, unsafeAddr t.data[0], bytes,
                    HippoMemcpyHostToDevice, stream)
  result = GpuTensor(
    devicePtr: alloc.p,
    alloc: alloc,
    shape: t.shape,
    sizeBytes: bytes,
  )

proc downloadToCpu*(gt: GpuTensor, stream: HippoStream): Tensor =
  ## Download a GPU tensor to a new CPU tensor.
  result = newTensor(gt.shape)
  hippoMemcpyAsync(addr result.data[0], gt.devicePtr, gt.sizeBytes,
                    HippoMemcpyDeviceToHost, stream)

proc copyToDevice*(dst: GpuTensor, src: Tensor, stream: HippoStream) =
  ## Copy CPU tensor data into an existing GPU tensor.
  let bytes = src.data.len * sizeof(float32)
  hippoMemcpyAsync(dst.devicePtr, unsafeAddr src.data[0], bytes,
                    HippoMemcpyHostToDevice, stream)

proc gpuMemcpyDevice*(dst, src: pointer, bytes: int, stream: HippoStream) =
  ## Device-to-device copy.
  hippoMemcpyAsync(dst, src, bytes, HippoMemcpyDeviceToDevice, stream)

var tokenAllocKeepAlive: HippoAllocRef  # prevent GC from freeing token buffer

proc gpuAllocAndUploadInt32*(data: pointer, count: int, stream: HippoStream): pointer =
  ## Allocate GPU memory and upload int32 data. Returns device pointer.
  let bytes = count * sizeof(int32)
  tokenAllocKeepAlive = hippoMalloc(bytes)
  hippoMemcpyAsync(tokenAllocKeepAlive.p, data, bytes, HippoMemcpyHostToDevice, stream)
  tokenAllocKeepAlive.p

proc gpuUploadToDevice*(dst: pointer, src: pointer, bytes: int, stream: HippoStream) =
  ## Upload host data to existing device pointer.
  hippoMemcpyAsync(dst, src, bytes, HippoMemcpyHostToDevice, stream)

proc gpuDownloadFromDevice*(dst: pointer, src: pointer, bytes: int, stream: HippoStream) =
  ## Download device data to host pointer.
  hippoMemcpyAsync(dst, src, bytes, HippoMemcpyDeviceToHost, stream)

proc gpuStreamSync*(stream: HippoStream) =
  ## Synchronize GPU stream.
  hippoStreamSynchronize(stream)

# ---------------------------------------------------------------------------
# GPU Context – single stream, weight cache, activation buffers
# ---------------------------------------------------------------------------
type
  GpuContext* = object
    initialized*: bool
    stream*: HippoStream
    weights*: Table[string, GpuTensor]
    # Ping-pong activation buffers
    act0*: GpuTensor
    act1*: GpuTensor
    actCapBytes*: int
    # Scratch buffers for intermediate results
    scratch0*: GpuTensor
    scratch1*: GpuTensor
    scratch2*: GpuTensor
    scratchCapBytes*: int

var gpuCtx*: GpuContext

proc ensureGpuContext*() =
  if not gpuCtx.initialized:
    gpuCtx.stream = hippoStreamCreate()
    gpuCtx.weights = initTable[string, GpuTensor]()
    gpuCtx.initialized = true

proc ensureActivationBuffers*(nElems: int) =
  ## Ensure ping-pong buffers are large enough.
  let bytes = nElems * sizeof(float32)
  if bytes > gpuCtx.actCapBytes:
    gpuCtx.act0 = newGpuTensor(@[nElems])
    gpuCtx.act1 = newGpuTensor(@[nElems])
    gpuCtx.actCapBytes = bytes

proc ensureScratchBuffers*(nElems: int) =
  ## Ensure scratch buffers are large enough.
  let bytes = nElems * sizeof(float32)
  if bytes > gpuCtx.scratchCapBytes:
    gpuCtx.scratch0 = newGpuTensor(@[nElems])
    gpuCtx.scratch1 = newGpuTensor(@[nElems])
    gpuCtx.scratch2 = newGpuTensor(@[nElems])
    gpuCtx.scratchCapBytes = bytes

proc cachedWeight*(name: string, w: Tensor): GpuTensor =
  ## Upload a named weight tensor to GPU once; return cached GpuTensor.
  ensureGpuContext()
  if gpuCtx.weights.hasKey(name):
    return gpuCtx.weights[name]
  let gt = uploadToGpu(w, gpuCtx.stream)
  gpuCtx.weights[name] = gt
  gt

proc cachedWeight*(w: Tensor): GpuTensor =
  ## Legacy fallback when a stable key is not available.
  ## This path avoids incorrect cache aliasing by skipping persistent cache.
  ensureGpuContext()
  let gt = uploadToGpu(w, gpuCtx.stream)
  gt

# ---------------------------------------------------------------------------
# GPU KV Cache
# ---------------------------------------------------------------------------
type
  GpuKvCache* = object
    k*: seq[GpuTensor]       # [kvDim, maxLen] per layer
    v*: seq[GpuTensor]       # [kvDim, maxLen] per layer
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int

proc initGpuKvCache*(nLayer, nHeadKv, headDim, maxLen: int): GpuKvCache =
  ensureGpuContext()
  let kvDim = nHeadKv * headDim
  result.maxLen = maxLen
  result.curLen = 0
  result.nHeadKv = nHeadKv
  result.headDim = headDim
  result.k = newSeq[GpuTensor](nLayer)
  result.v = newSeq[GpuTensor](nLayer)
  for i in 0 ..< nLayer:
    result.k[i] = newGpuTensor(@[kvDim, maxLen])
    result.v[i] = newGpuTensor(@[kvDim, maxLen])

# ---------------------------------------------------------------------------
# Kernel: Elementwise Add (residual connection)
# ---------------------------------------------------------------------------
proc addKernel(aData, bData, outData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let a = cast[ptr UncheckedArray[float32]](aData)
    let b = cast[ptr UncheckedArray[float32]](bData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    o[idx] = a[idx] + b[idx]

proc gpuAdd*(dst: pointer, a, b: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var aPtr = a
  var bPtr = b
  var dPtr = dst
  var n = nElems.cint
  hippoLaunchKernel(addKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(aPtr, bPtr, dPtr, n))

# ---------------------------------------------------------------------------
# Kernel: Fused SiLU * elementwise multiply (for FFN gate)
# ---------------------------------------------------------------------------
proc siluMulKernel(gateData, upData, outData: ptr float32, n: cint) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  if idx < int(n):
    let g = cast[ptr UncheckedArray[float32]](gateData)
    let u = cast[ptr UncheckedArray[float32]](upData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    let x = g[idx]
    let sigmoid = 1.0'f32 / (1.0'f32 + expf(-x))
    o[idx] = x * sigmoid * u[idx]

proc gpuSiluMul*(dst: pointer, gate, up: pointer, nElems: int, stream: HippoStream) =
  let grid = newDim3(((nElems + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var gPtr = gate
  var uPtr = up
  var dPtr = dst
  var n = nElems.cint
  hippoLaunchKernel(siluMulKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(gPtr, uPtr, dPtr, n))

# ---------------------------------------------------------------------------
# Kernel: RMSNorm over columns (ggml layout: [dim, seqLen])
# Each block handles one column. Shared memory for reduction.
# ---------------------------------------------------------------------------
proc rmsnormColsKernel(
  xData, weightData, outData: ptr float32,
  dim, seqLen: cint, eps: float32
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let col = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if col >= int(seqLen):
    return

  let x = cast[ptr UncheckedArray[float32]](xData)
  let w = cast[ptr UncheckedArray[float32]](weightData)
  let o = cast[ptr UncheckedArray[float32]](outData)

  # Each thread accumulates sum-of-squares for its chunk of rows
  var ss = 0.0'f32
  var r = tid
  while r < int(dim):
    let v = x[r * int(seqLen) + col]
    ss = ss + v * v
    r = r + int(blockDim.x)
  sdata[tid] = ss
  hippoSyncthreads()

  # Tree reduction in shared memory
  var stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sdata[tid] = sdata[tid] + sdata[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  # Broadcast inv
  var inv {.hippoShared.}: array[1, float32]
  if tid == 0:
    inv[0] = 1.0'f32 / sqrtf(sdata[0] / cfloat(dim) + cfloat(eps))
  hippoSyncthreads()

  let invVal = inv[0]
  r = tid
  while r < int(dim):
    let idx = r * int(seqLen) + col
    o[idx] = x[idx] * invVal * w[r]
    r = r + int(blockDim.x)

proc gpuRmsnormCols*(dst: pointer, x, weight: pointer, dim, seqLen: int,
                      eps: float32, stream: HippoStream) =
  let grid = newDim3(seqLen.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x
  var wPtr = weight
  var dPtr = dst
  var dimArg = dim.cint
  var seqLenArg = seqLen.cint
  var epsArg = eps
  hippoLaunchKernel(rmsnormColsKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, wPtr, dPtr, dimArg, seqLenArg, epsArg))

# ---------------------------------------------------------------------------
# Kernel: Embedding lookup
# ---------------------------------------------------------------------------
proc embeddingKernel(
  weightData, outData: ptr float32,
  tokenIds: ptr int32,
  nEmb, nTokens, nVocab: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let totalElems = int(nEmb) * int(nTokens)
  if idx < totalElems:
    let e = idx div int(nTokens)  # which embedding dimension
    let t = idx mod int(nTokens)  # which token position
    let w = cast[ptr UncheckedArray[float32]](weightData)
    let o = cast[ptr UncheckedArray[float32]](outData)
    let toks = cast[ptr UncheckedArray[int32]](tokenIds)
    let tid = int(toks[t])
    # weight layout: [nVocab/nEmb, nEmb/nVocab] - use ggml layout (rows=nEmb)
    # weight[tid * nEmb + e] -> out[e * nTokens + t]
    o[e * int(nTokens) + t] = w[tid * int(nEmb) + e]

proc gpuEmbedding*(dst: pointer, weight: pointer, tokenIds: ptr int32,
                    nEmb, nTokens, nVocab: int, stream: HippoStream) =
  let total = nEmb * nTokens
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var wPtr = weight
  var dPtr = dst
  var tPtr = cast[pointer](tokenIds)
  var nEmbArg = nEmb.cint
  var nTokArg = nTokens.cint
  var nVocArg = nVocab.cint
  hippoLaunchKernel(embeddingKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(wPtr, dPtr, tPtr, nEmbArg, nTokArg, nVocArg))

# ---------------------------------------------------------------------------
# Kernel: RoPE (rotary position embeddings)
# For decode (seqLen=1): each thread handles one (head, pair) combo
# ---------------------------------------------------------------------------
proc ropeAtPosKernel(
  xData: ptr float32,
  nHead, headDim, ropeDim: cint,
  ropeBase: float32, pos: cint, seqLen: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let halfRope = int(ropeDim) div 2
  let totalPairs = int(nHead) * halfRope
  if idx >= totalPairs:
    return
  let x = cast[ptr UncheckedArray[float32]](xData)
  let h = idx div halfRope
  let i = idx mod halfRope
  let hOffset = h * int(headDim)

  let theta = powf(1.0'f32 / cfloat(ropeBase), cfloat(2 * i) / cfloat(ropeDim))
  let angle = cfloat(pos) * theta
  let c = cosf(angle)
  let s = sinf(angle)

  # For seqLen==1 decode case
  if int(seqLen) == 1:
    let idx0 = hOffset + 2 * i
    let idx1 = hOffset + 2 * i + 1
    let v0 = x[idx0]
    let v1 = x[idx1]
    x[idx0] = v0 * c - v1 * s
    x[idx1] = v0 * s + v1 * c
  else:
    # Prefill case: each thread iterates over positions
    # We'll launch enough threads for nHead * halfRope
    # and iterate over seqLen positions
    var p = 0
    while p < int(seqLen):
      let idx0 = (hOffset + 2 * i) * int(seqLen) + p
      let idx1 = (hOffset + 2 * i + 1) * int(seqLen) + p
      let pTheta = powf(1.0'f32 / cfloat(ropeBase), cfloat(2 * i) / cfloat(ropeDim))
      let pAngle = cfloat(p) * pTheta
      let pc = cosf(pAngle)
      let ps = sinf(pAngle)
      let v0 = x[idx0]
      let v1 = x[idx1]
      x[idx0] = v0 * pc - v1 * ps
      x[idx1] = v0 * ps + v1 * pc
      p = p + 1

proc gpuRopeAtPos*(x: pointer, nHead, headDim, ropeDim: int,
                    ropeBase: float32, pos: int, seqLen: int,
                    stream: HippoStream) =
  let halfRope = ropeDim div 2
  let totalPairs = nHead * halfRope
  let grid = newDim3(((totalPairs + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var xPtr = x
  var nHeadArg = nHead.cint
  var headDimArg = headDim.cint
  var ropeDimArg = ropeDim.cint
  var ropeBaseArg = ropeBase
  var posArg = pos.cint
  var seqLenArg = seqLen.cint
  hippoLaunchKernel(ropeAtPosKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(xPtr, nHeadArg, headDimArg, ropeDimArg,
                                     ropeBaseArg, posArg, seqLenArg))

# ---------------------------------------------------------------------------
# Kernel: Store KV (write k,v vectors into KV cache)
# ---------------------------------------------------------------------------
proc storeKVKernel(
  cacheData, srcData: ptr float32,
  rows, srcCols, cacheCols, startPos: cint
) {.hippoGlobal.} =
  let idx = int(blockIdx.x * blockDim.x + threadIdx.x)
  let totalElems = int(rows) * int(srcCols)
  if idx < totalElems:
    let r = idx div int(srcCols)
    let c = idx mod int(srcCols)
    let cache = cast[ptr UncheckedArray[float32]](cacheData)
    let src = cast[ptr UncheckedArray[float32]](srcData)
    cache[r * int(cacheCols) + int(startPos) + c] = src[r * int(srcCols) + c]

proc gpuStoreKV*(cache: pointer, src: pointer,
                  rows, srcCols, cacheCols, startPos: int,
                  stream: HippoStream) =
  let total = rows * srcCols
  let grid = newDim3(((total + HippoBlockSize - 1) div HippoBlockSize).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var cPtr = cache
  var sPtr = src
  var rowsArg = rows.cint
  var srcColsArg = srcCols.cint
  var cacheColsArg = cacheCols.cint
  var startPosArg = startPos.cint
  hippoLaunchKernel(storeKVKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(cPtr, sPtr, rowsArg, srcColsArg,
                                     cacheColsArg, startPosArg))

# ---------------------------------------------------------------------------
# Kernel: Attention decode (single-token, cached KV)
# One block per head. Shared memory for scores + softmax.
# ---------------------------------------------------------------------------
proc attentionDecodeKernel(
  qData, kCacheData, vCacheData, outData: ptr float32,
  nHead, nHeadKv, headDim, curLen, cacheCols: cint,
  invSqrtHeadDim: float32
) {.hippoGlobal.} =
  # One block per head
  let h = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return

  let q = cast[ptr UncheckedArray[float32]](qData)
  let kc = cast[ptr UncheckedArray[float32]](kCacheData)
  let vc = cast[ptr UncheckedArray[float32]](vCacheData)
  let o = cast[ptr UncheckedArray[float32]](outData)

  let group = int(nHead) div int(nHeadKv)
  let kvh = h div group
  let hOff = h * int(headDim)

  # Phase 1: compute attention scores - each thread handles some positions
  # We use shared memory for partial max/sum
  var scores {.hippoShared.}: array[4096, float32]  # max curLen we support per block
  var sMax {.hippoShared.}: array[HippoBlockSize, float32]
  var sSum {.hippoShared.}: array[HippoBlockSize, float32]

  # Each thread computes scores for its positions
  var localMax = -1e30'f32
  var j = tid
  while j < int(curLen):
    var dot = 0.0'f32
    for d in 0 ..< int(headDim):
      let qIdx = hOff + d
      let kIdx = (kvh * int(headDim) + d) * int(cacheCols) + j
      dot = dot + q[qIdx] * kc[kIdx]
    let score = dot * invSqrtHeadDim
    scores[j] = score
    if score > localMax:
      localMax = score
    j = j + int(blockDim.x)
  sMax[tid] = localMax
  hippoSyncthreads()

  # Reduce max across threads
  var stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      if sMax[tid + stride] > sMax[tid]:
        sMax[tid] = sMax[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  let globalMax = sMax[0]

  # Exp and partial sum
  var localSum = 0.0'f32
  j = tid
  while j < int(curLen):
    let e = expf(scores[j] - globalMax)
    scores[j] = e
    localSum = localSum + e
    j = j + int(blockDim.x)
  sSum[tid] = localSum
  hippoSyncthreads()

  # Reduce sum
  stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sSum[tid] = sSum[tid] + sSum[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  let invSum = 1.0'f32 / sSum[0]

  # Normalize scores
  j = tid
  while j < int(curLen):
    scores[j] = scores[j] * invSum
    j = j + int(blockDim.x)
  hippoSyncthreads()

  # Phase 2: weighted sum of V
  var d = tid
  while d < int(headDim):
    var acc = 0.0'f32
    for jj in 0 ..< int(curLen):
      let vIdx = (kvh * int(headDim) + d) * int(cacheCols) + jj
      acc = acc + scores[jj] * vc[vIdx]
    let outIdx = hOff + d
    o[outIdx] = acc
    d = d + int(blockDim.x)

proc gpuAttentionDecode*(dst: pointer, q, kCache, vCache: pointer,
                          nHead, nHeadKv, headDim, curLen, cacheCols: int,
                          stream: HippoStream) =
  let grid = newDim3(nHead.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  let invSqrt = 1.0'f32 / sqrtf(headDim.float32)
  var qPtr = q
  var kcPtr = kCache
  var vcPtr = vCache
  var dPtr = dst
  var nHeadArg = nHead.cint
  var nHeadKvArg = nHeadKv.cint
  var headDimArg = headDim.cint
  var curLenArg = curLen.cint
  var cacheColsArg = cacheCols.cint
  var invSqrtArg = invSqrt
  hippoLaunchKernel(attentionDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kcPtr, vcPtr, dPtr,
                                     nHeadArg, nHeadKvArg, headDimArg,
                                     curLenArg, cacheColsArg, invSqrtArg))

# ---------------------------------------------------------------------------
# Kernel: Attention prefill (multi-token, causal masking)
# One block per (head, query-position) pair
# ---------------------------------------------------------------------------
proc attentionPrefillKernel(
  qData, kData, vData, outData: ptr float32,
  nHead, nHeadKv, headDim, seqLen: cint,
  invSqrtHeadDim: float32
) {.hippoGlobal.} =
  let blockId = int(blockIdx.x)
  let h = blockId div int(seqLen)
  let qi = blockId mod int(seqLen)
  let tid = int(threadIdx.x)
  if h >= int(nHead):
    return

  let qArr = cast[ptr UncheckedArray[float32]](qData)
  let kArr = cast[ptr UncheckedArray[float32]](kData)
  let vArr = cast[ptr UncheckedArray[float32]](vData)
  let oArr = cast[ptr UncheckedArray[float32]](outData)

  let group = int(nHead) div int(nHeadKv)
  let kvh = h div group
  let hOff = h * int(headDim)
  let causalLen = qi + 1  # only attend to positions 0..qi

  var scores {.hippoShared.}: array[4096, float32]
  var sMax {.hippoShared.}: array[HippoBlockSize, float32]
  var sSum {.hippoShared.}: array[HippoBlockSize, float32]

  # Compute attention scores
  var localMax = -1e30'f32
  var j = tid
  while j < causalLen:
    var dot = 0.0'f32
    for d in 0 ..< int(headDim):
      let qIdx = (hOff + d) * int(seqLen) + qi
      let kIdx = (kvh * int(headDim) + d) * int(seqLen) + j
      dot = dot + qArr[qIdx] * kArr[kIdx]
    let score = dot * invSqrtHeadDim
    scores[j] = score
    if score > localMax:
      localMax = score
    j = j + int(blockDim.x)
  sMax[tid] = localMax
  hippoSyncthreads()

  var stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      if sMax[tid + stride] > sMax[tid]:
        sMax[tid] = sMax[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  let globalMax = sMax[0]

  var localSum = 0.0'f32
  j = tid
  while j < causalLen:
    let e = expf(scores[j] - globalMax)
    scores[j] = e
    localSum = localSum + e
    j = j + int(blockDim.x)
  sSum[tid] = localSum
  hippoSyncthreads()

  stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sSum[tid] = sSum[tid] + sSum[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  let invSum = 1.0'f32 / sSum[0]

  j = tid
  while j < causalLen:
    scores[j] = scores[j] * invSum
    j = j + int(blockDim.x)
  hippoSyncthreads()

  # Weighted sum of V
  var d = tid
  while d < int(headDim):
    var acc = 0.0'f32
    for jj in 0 ..< causalLen:
      let vIdx = (kvh * int(headDim) + d) * int(seqLen) + jj
      acc = acc + scores[jj] * vArr[vIdx]
    let outIdx = (hOff + d) * int(seqLen) + qi
    oArr[outIdx] = acc
    d = d + int(blockDim.x)

proc gpuAttentionPrefill*(dst: pointer, q, k, v: pointer,
                           nHead, nHeadKv, headDim, seqLen: int,
                           stream: HippoStream) =
  let nBlocks = nHead * seqLen
  let grid = newDim3(nBlocks.uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  let invSqrt = 1.0'f32 / sqrtf(headDim.float32)
  var qPtr = q
  var kPtr = k
  var vPtr = v
  var dPtr = dst
  var nHeadArg = nHead.cint
  var nHeadKvArg = nHeadKv.cint
  var headDimArg = headDim.cint
  var seqLenArg = seqLen.cint
  var invSqrtArg = invSqrt
  hippoLaunchKernel(attentionPrefillKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(qPtr, kPtr, vPtr, dPtr,
                                     nHeadArg, nHeadKvArg, headDimArg,
                                     seqLenArg, invSqrtArg))

# ---------------------------------------------------------------------------
# Kernel: Naive GEMM (baseline, kept for prefill)
# ---------------------------------------------------------------------------
proc linearHippoKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols, seqLen: cint
) {.hippoGlobal.} =
  let outRow = int(blockIdx.y * blockDim.y + threadIdx.y)
  let seqCol = int(blockIdx.x * blockDim.x + threadIdx.x)
  if outRow < int(outRows) and seqCol < int(seqLen):
    let wArray = cast[ptr UncheckedArray[float32]](wData)
    let xArray = cast[ptr UncheckedArray[float32]](xData)
    let outArray = cast[ptr UncheckedArray[float32]](outData)
    var acc = 0.0'f32
    for k in 0 ..< int(wCols):
      acc = acc + wArray[outRow * int(wCols) + k] * xArray[k * int(seqLen) + seqCol]
    outArray[outRow * int(seqLen) + seqCol] = acc

# ---------------------------------------------------------------------------
# Kernel: Parallel GEMV decode (one block per output row)
# Opt-1 implementation: many threads cooperate on each dot product
# ---------------------------------------------------------------------------
proc linearHippoDecodeKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let outRow = int(blockIdx.x)
  let tid = int(threadIdx.x)
  if outRow >= int(outRows):
    return

  let wArray = cast[ptr UncheckedArray[float32]](wData)
  let xArray = cast[ptr UncheckedArray[float32]](xData)
  let outArray = cast[ptr UncheckedArray[float32]](outData)
  let rowBase = outRow * int(wCols)

  var acc = 0.0'f32
  var k = tid
  while k < int(wCols):
    acc = acc + wArray[rowBase + k] * xArray[k]
    k = k + int(blockDim.x)
  sdata[tid] = acc
  hippoSyncthreads()

  var stride = int(blockDim.x) div 2
  while stride > 0:
    if tid < stride:
      sdata[tid] = sdata[tid] + sdata[tid + stride]
    hippoSyncthreads()
    stride = stride div 2

  if tid == 0:
    outArray[outRow] = sdata[0]

# ---------------------------------------------------------------------------
# Device GEMM dispatcher (operates on device pointers, no CPU roundtrip)
# ---------------------------------------------------------------------------
proc gpuLinearCol*(dst, x, w: pointer, wCols, wRows, seqLen: int,
                   stream: HippoStream) =
  var wPtr = w
  var xPtr = x
  var dPtr = dst
  var outRowsArg = wRows.cint
  var wColsArg = wCols.cint

  if seqLen == 1:
    # Optimized decode path: one block per output row, parallel reduction.
    let grid = newDim3(wRows.uint32)
    let blk = newDim3(HippoBlockSize.uint32)
    hippoLaunchKernel(linearHippoDecodeKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))
  else:
    # Naive GEMM: one thread per output element
    let gridX = (seqLen + HippoBlockSizeX - 1) div HippoBlockSizeX
    let gridY = (wRows + HippoBlockSizeY - 1) div HippoBlockSizeY
    let grid = newDim3(gridX.uint32, gridY.uint32)
    let blk = newDim3(HippoBlockSizeX.uint32, HippoBlockSizeY.uint32)
    var seqLenArg = seqLen.cint
    hippoLaunchKernel(linearHippoKernel, gridDim = grid, blockDim = blk,
                      stream = stream,
                      args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg, seqLenArg))

# ---------------------------------------------------------------------------
# Legacy API: linearHippoCol (CPU tensor in/out, for backward compat)
# ---------------------------------------------------------------------------

proc linearHippoCol*(x: Tensor, w: Tensor, wCols, wRows, seqLen: int): Tensor =
  result = newTensor(@[wRows, seqLen])
  if result.data.len == 0:
    return result

  ensureGpuContext()
  let stream = gpuCtx.stream

  let xBytes = x.data.len * sizeof(float32)
  let outBytes = result.data.len * sizeof(float32)

  ensureActivationBuffers(max(x.data.len, result.data.len))

  let devW = cachedWeight(w)
  let devX = gpuCtx.act0.devicePtr
  let devOut = gpuCtx.act1.devicePtr

  hippoMemcpyAsync(devX, unsafeAddr x.data[0], xBytes,
                    HippoMemcpyHostToDevice, stream)

  gpuLinearCol(devOut, devX, devW.devicePtr, wCols, wRows, seqLen, stream)

  hippoMemcpyAsync(addr result.data[0], devOut, outBytes,
                    HippoMemcpyDeviceToHost, stream)
  hippoStreamSynchronize(stream)
