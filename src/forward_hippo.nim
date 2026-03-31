## Hippo GPU backend: kernels and GPU-resident forward pass.
##
## All model weights, activations, and KV caches live on GPU.
## Only the initial token IDs are uploaded and final logits downloaded.

import
  std/tables,
  ./[tensor, model, gguf_loader, quant]

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
  HippoDecodeRowsPerBlock = 4
  HippoDecodeDotUnroll = 4
  HippoMaxDecodeCols = 5632

when HippoDecodeDotUnroll != 4:
  {.error: "linearHippoDecodeKernel currently implements a fixed 4-way unroll.".}

template reduceSum256(sdata: var array[HippoBlockSize, float32], tid: int) =
  ## 256-thread tree reduction with warp shuffle for the final intra-warp steps.
  when HippoBlockSize == 256:
    if tid < 128:
      sdata[tid] = sdata[tid] + sdata[tid + 128]
    hippoSyncthreads()
    when HippoWarpSize == 64:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        val = val + hippoShflDown(val, 32)
        val = val + hippoShflDown(val, 16)
        val = val + hippoShflDown(val, 8)
        val = val + hippoShflDown(val, 4)
        val = val + hippoShflDown(val, 2)
        val = val + hippoShflDown(val, 1)
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    elif HippoWarpSize == 32:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        sdata[tid] = sdata[tid] + sdata[tid + 32]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        val = val + hippoShflDown(val, 16)
        val = val + hippoShflDown(val, 8)
        val = val + hippoShflDown(val, 4)
        val = val + hippoShflDown(val, 2)
        val = val + hippoShflDown(val, 1)
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    else:
      if tid < 64:
        sdata[tid] = sdata[tid] + sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        sdata[tid] = sdata[tid] + sdata[tid + 32]
      hippoSyncthreads()
      if tid < 16:
        sdata[tid] = sdata[tid] + sdata[tid + 16]
      hippoSyncthreads()
      if tid < 8:
        sdata[tid] = sdata[tid] + sdata[tid + 8]
      hippoSyncthreads()
      if tid < 4:
        sdata[tid] = sdata[tid] + sdata[tid + 4]
      hippoSyncthreads()
      if tid < 2:
        sdata[tid] = sdata[tid] + sdata[tid + 2]
      hippoSyncthreads()
      if tid < 1:
        sdata[tid] = sdata[tid] + sdata[tid + 1]
      hippoSyncthreads()

template reduceMax256(sdata: var array[HippoBlockSize, float32], tid: int) =
  ## 256-thread tree max-reduction with warp shuffle for the final intra-warp steps.
  when HippoBlockSize == 256:
    if tid < 128:
      if sdata[tid + 128] > sdata[tid]:
        sdata[tid] = sdata[tid + 128]
    hippoSyncthreads()
    when HippoWarpSize == 64:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        var other = hippoShflDown(val, 32)
        if other > val: val = other
        other = hippoShflDown(val, 16)
        if other > val: val = other
        other = hippoShflDown(val, 8)
        if other > val: val = other
        other = hippoShflDown(val, 4)
        if other > val: val = other
        other = hippoShflDown(val, 2)
        if other > val: val = other
        other = hippoShflDown(val, 1)
        if other > val: val = other
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    elif HippoWarpSize == 32:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        if sdata[tid + 32] > sdata[tid]:
          sdata[tid] = sdata[tid + 32]
      hippoSyncthreads()
      if tid < HippoWarpSize:
        var val = sdata[tid]
        var other = hippoShflDown(val, 16)
        if other > val: val = other
        other = hippoShflDown(val, 8)
        if other > val: val = other
        other = hippoShflDown(val, 4)
        if other > val: val = other
        other = hippoShflDown(val, 2)
        if other > val: val = other
        other = hippoShflDown(val, 1)
        if other > val: val = other
        if tid == 0:
          sdata[0] = val
      hippoSyncthreads()
    else:
      if tid < 64:
        if sdata[tid + 64] > sdata[tid]:
          sdata[tid] = sdata[tid + 64]
      hippoSyncthreads()
      if tid < 32:
        if sdata[tid + 32] > sdata[tid]:
          sdata[tid] = sdata[tid + 32]
      hippoSyncthreads()
      if tid < 16:
        if sdata[tid + 16] > sdata[tid]:
          sdata[tid] = sdata[tid + 16]
      hippoSyncthreads()
      if tid < 8:
        if sdata[tid + 8] > sdata[tid]:
          sdata[tid] = sdata[tid + 8]
      hippoSyncthreads()
      if tid < 4:
        if sdata[tid + 4] > sdata[tid]:
          sdata[tid] = sdata[tid + 4]
      hippoSyncthreads()
      if tid < 2:
        if sdata[tid + 2] > sdata[tid]:
          sdata[tid] = sdata[tid + 2]
      hippoSyncthreads()
      if tid < 1:
        if sdata[tid + 1] > sdata[tid]:
          sdata[tid] = sdata[tid + 1]
      hippoSyncthreads()

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

var tokenBufAlloc: HippoAllocRef
var tokenBufCapacity: int = 0

proc gpuUploadInt32Pooled*(data: pointer, count: int, stream: HippoStream): pointer =
  ## Upload int32 data to a reusable GPU buffer (no hipMalloc per call).
  let bytes = count * sizeof(int32)
  if bytes > tokenBufCapacity:
    tokenBufAlloc = hippoMalloc(max(bytes, 16))
    tokenBufCapacity = max(bytes, 16)
  hippoMemcpyAsync(tokenBufAlloc.p, data, bytes, HippoMemcpyHostToDevice, stream)
  tokenBufAlloc.p

# ---------------------------------------------------------------------------
# Pre-computed layer GPU pointers — zero string ops / table lookups in hot loop
# ---------------------------------------------------------------------------
type
  LayerGpuPtrs* = object
    attnNorm*, ffnNorm*: pointer          # always float32 (small 1D tensors)
    wq*, wk*, wv*, wo*: pointer            # float32 device pointers (nil if quantized)
    wGate*, wUp*, wDown*: pointer
    wqQ*, wkQ*, wvQ*, woQ*: pointer        # quantized device pointers (nil if float32)
    wGateQ*, wUpQ*, wDownQ*: pointer
    wColsQ*, wColsDown*: int               # column counts for quant dispatch
    wqQType*, wkQType*, wvQType*, woQType*: int32
    wGateQType*, wUpQType*, wDownQType*: int32

  ModelGpuPtrs* = object
    layers*: seq[LayerGpuPtrs]
    tokEmb*: pointer
    normWeight*, outputWeight*: pointer
    outputShape0*, outputShape1*: int
    initialized*: bool

var modelPtrs*: ModelGpuPtrs

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
# GpuQuantWeight — raw quantized bytes on GPU (no dequant to float32)
# ---------------------------------------------------------------------------
type
  GpuQuantWeight* = object
    devicePtr*: pointer
    alloc: HippoAllocRef
    sizeBytes*: int
    nRows*: int
    nCols*: int           # number of float elements per row

var quantWeightCache: Table[string, GpuQuantWeight]

proc cachedQuantWeight*(name: string, m: var Model, tensorName: string): GpuQuantWeight =
  if quantWeightCache.hasKey(name):
    return quantWeightCache[name]
  ensureGpuContext()
  let info = m.infos[tensorName]
  let dataPtr = tensorDataPtr(m.gguf, info)
  let nCols = int(info.ne[0])
  let nRows = tensorElemCount(info) div nCols
  let rowSize = case info.elemType
    of GgmlTypeQ2K: rowSizeQ2K(nCols)
    of GgmlTypeQ3K: rowSizeQ3K(nCols)
    of GgmlTypeQ6K: rowSizeQ6K(nCols)
    else: raise newException(ValueError, "unsupported quant type for GPU upload: " & $info.elemType)
  let totalBytes = rowSize * nRows
  let alloc = hippoMalloc(totalBytes)
  hippoMemcpyAsync(alloc.p, dataPtr, totalBytes, HippoMemcpyHostToDevice, gpuCtx.stream)
  result = GpuQuantWeight(devicePtr: alloc.p, alloc: alloc, sizeBytes: totalBytes,
                          nRows: nRows, nCols: nCols)
  quantWeightCache[name] = result


# ---------------------------------------------------------------------------
# Kernel: Q2_K GEMV decode — reads raw quantized bytes, dequants on the fly
# ---------------------------------------------------------------------------
proc linearQ2KDecodeKernel(
  wData: ptr uint8,          # raw Q2_K bytes
  xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sx {.hippoShared.}: array[HippoMaxDecodeCols, float32]
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let blockSize = int(blockDim.x)
  let cols = int(wCols)
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let w = cast[ptr UncheckedArray[uint8]](wData)
  let xArr = cast[ptr UncheckedArray[float32]](xData)
  let outArr = cast[ptr UncheckedArray[float32]](outData)
  let nBlocksPerRow = cols div 256
  let rowSizeBytes = nBlocksPerRow * 84

  # Load input vector into shared memory
  var k = tid
  while k < cols:
    sx[k] = xArr[k]
    k = k + blockSize
  hippoSyncthreads()

  # Pre-compute element mapping — constant across all blocks and rows
  let q2_chunk = tid shr 7
  let q2_localElem = tid and 127
  let q2_iteration = q2_localElem shr 5
  let q2_sub = (q2_localElem shr 4) and 1
  let q2_l = q2_localElem and 15
  let q2_scaleIdx = q2_chunk * 8 + q2_iteration * 2 + q2_sub
  let q2_shift = q2_iteration * 2
  let q2_qsByteIdx = q2_chunk * 32 + q2_sub * 16 + q2_l

  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r
    if outRow < int(outRows):
      let rowBase = outRow * rowSizeBytes
      var acc = 0.0'f32

      var blkIdx = 0
      # 4-way block unroll for ILP
      while blkIdx + 3 < nBlocksPerRow:
        # Block A
        let blkStartA = rowBase + blkIdx * 84
        let elemBaseA = blkIdx * 256
        let dRawA = uint16(w[blkStartA + 80]) or (uint16(w[blkStartA + 81]) shl 8)
        let dminRawA = uint16(w[blkStartA + 82]) or (uint16(w[blkStartA + 83]) shl 8)
        let dA = hippoHalfToFloat(dRawA)
        let dminA = hippoHalfToFloat(dminRawA)
        let scA = w[blkStartA + q2_scaleIdx]
        let dlA = dA * cfloat(scA and 0x0F'u8)
        let mlA = dminA * cfloat(scA shr 4)
        let qvalA = cfloat((w[blkStartA + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlA * qvalA - mlA) * sx[elemBaseA + tid]

        # Block B
        let blkStartB = rowBase + (blkIdx + 1) * 84
        let elemBaseB = (blkIdx + 1) * 256
        let dRawB = uint16(w[blkStartB + 80]) or (uint16(w[blkStartB + 81]) shl 8)
        let dminRawB = uint16(w[blkStartB + 82]) or (uint16(w[blkStartB + 83]) shl 8)
        let dB = hippoHalfToFloat(dRawB)
        let dminB = hippoHalfToFloat(dminRawB)
        let scB = w[blkStartB + q2_scaleIdx]
        let dlB = dB * cfloat(scB and 0x0F'u8)
        let mlB = dminB * cfloat(scB shr 4)
        let qvalB = cfloat((w[blkStartB + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlB * qvalB - mlB) * sx[elemBaseB + tid]

        # Block C
        let blkStartC = rowBase + (blkIdx + 2) * 84
        let elemBaseC = (blkIdx + 2) * 256
        let dRawC = uint16(w[blkStartC + 80]) or (uint16(w[blkStartC + 81]) shl 8)
        let dminRawC = uint16(w[blkStartC + 82]) or (uint16(w[blkStartC + 83]) shl 8)
        let dC = hippoHalfToFloat(dRawC)
        let dminC = hippoHalfToFloat(dminRawC)
        let scC = w[blkStartC + q2_scaleIdx]
        let dlC = dC * cfloat(scC and 0x0F'u8)
        let mlC = dminC * cfloat(scC shr 4)
        let qvalC = cfloat((w[blkStartC + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlC * qvalC - mlC) * sx[elemBaseC + tid]

        # Block D
        let blkStartD = rowBase + (blkIdx + 3) * 84
        let elemBaseD = (blkIdx + 3) * 256
        let dRawD = uint16(w[blkStartD + 80]) or (uint16(w[blkStartD + 81]) shl 8)
        let dminRawD = uint16(w[blkStartD + 82]) or (uint16(w[blkStartD + 83]) shl 8)
        let dD = hippoHalfToFloat(dRawD)
        let dminD = hippoHalfToFloat(dminRawD)
        let scD = w[blkStartD + q2_scaleIdx]
        let dlD = dD * cfloat(scD and 0x0F'u8)
        let mlD = dminD * cfloat(scD shr 4)
        let qvalD = cfloat((w[blkStartD + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dlD * qvalD - mlD) * sx[elemBaseD + tid]

        blkIdx = blkIdx + 4

      # Cleanup: remaining blocks
      while blkIdx < nBlocksPerRow:
        let blkStart = rowBase + blkIdx * 84
        let elemBase = blkIdx * 256
        let dRaw = uint16(w[blkStart + 80]) or (uint16(w[blkStart + 81]) shl 8)
        let dminRaw = uint16(w[blkStart + 82]) or (uint16(w[blkStart + 83]) shl 8)
        let d = hippoHalfToFloat(dRaw)
        let dmin = hippoHalfToFloat(dminRaw)
        let sc = w[blkStart + q2_scaleIdx]
        let dl = d * cfloat(sc and 0x0F'u8)
        let ml = dmin * cfloat(sc shr 4)
        let qval = cfloat((w[blkStart + 16 + q2_qsByteIdx] shr q2_shift) and 3)
        acc = acc + (dl * qval - ml) * sx[elemBase + tid]
        blkIdx = blkIdx + 1

      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()

    reduceSum256(sdata, tid)

    if tid == 0 and outRow < int(outRows):
      outArr[outRow] = sdata[0]
    hippoSyncthreads()

proc gpuLinearColQ2K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  if wCols > HippoMaxDecodeCols:
    raise newException(ValueError, "Q2K decode width exceeds limit: " & $wCols)
  let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var wPtr = wQuant; var xPtr = x; var dPtr = dst
  var outRowsArg = wRows.cint; var wColsArg = wCols.cint
  hippoLaunchKernel(linearQ2KDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))

# ---------------------------------------------------------------------------
# Kernel: Q3_K GEMV decode — reads raw quantized bytes, dequants on the fly
# ---------------------------------------------------------------------------
# Q3_K block layout (110 bytes → 256 float32 elements):
#   hmask[32]   @ offset 0   — 1 high-bit per element (256 bits)
#   qs[64]      @ offset 32  — 2 low bits per element, 4 packed per byte
#   scales[12]  @ offset 96  — packed scale bytes (decoded into 16 x int8)
#   d           @ offset 108 — float16 block scale
proc linearQ3KDecodeKernel(
  wData: ptr uint8,
  xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sx {.hippoShared.}: array[HippoMaxDecodeCols, float32]
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let blockSize = int(blockDim.x)
  let cols = int(wCols)
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let w = cast[ptr UncheckedArray[uint8]](wData)
  let xArr = cast[ptr UncheckedArray[float32]](xData)
  let outArr = cast[ptr UncheckedArray[float32]](outData)
  let nBlocksPerRow = cols div 256
  let rowSizeBytes = nBlocksPerRow * 110  # blockQ3KSize = 110

  # Load input vector into shared memory
  var k = tid
  while k < cols:
    sx[k] = xArr[k]
    k = k + blockSize
  hippoSyncthreads()

  # Pre-compute element mapping — constant across all blocks and rows
  let chunk = tid shr 7              # 0 or 1
  let localElem = tid and 127
  let iteration = localElem shr 5    # 0-3
  let sub = (localElem shr 4) and 1  # 0 or 1
  let l = localElem and 15           # 0-15
  let scaleIdx = chunk * 8 + iteration * 2 + sub
  let shift = iteration * 2
  let qsByteIdx = chunk * 32 + sub * 16 + l
  let hmaskByteOff = qsByteIdx mod 32  # = sub*16 + l
  let hmaskBitPos = chunk * 4 + iteration
  # Scale extraction offsets
  let byteInGroup = scaleIdx and 3
  let auxIdx = scaleIdx shr 2
  let sByteOff = 96 + (auxIdx and 1) * 4 + byteInGroup
  let tByteOff = 96 + 8 + byteInGroup
  let useHighShift = (auxIdx shr 1) * 4
  let auxShift = auxIdx * 2

  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r
    if outRow < int(outRows):
      let rowBase = outRow * rowSizeBytes
      var acc = 0.0'f32

      var blkIdx = 0
      # 4-way block unroll for ILP
      while blkIdx + 3 < nBlocksPerRow:
        # Block A
        let blkStartA = rowBase + blkIdx * 110
        let elemBaseA = blkIdx * 256
        let dRawA = uint16(w[blkStartA + 108]) or (uint16(w[blkStartA + 109]) shl 8)
        let dAllA = hippoHalfToFloat(dRawA)
        let sByteA = cint(w[blkStartA + sByteOff])
        let tByteA = cint(w[blkStartA + tByteOff])
        let lowA = (sByteA shr useHighShift) and 0x0F
        let highA = ((tByteA shr auxShift) and 0x03) shl 4
        let scaleByteA = lowA or highA
        let scaleSignedA = ((scaleByteA xor 0x80) - 0x80)
        let dlA = dAllA * cfloat(scaleSignedA - 32)
        let qvalA = cint((w[blkStartA + 32 + qsByteIdx] shr shift) and 3)
        let hmBitA = (cint(w[blkStartA + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmA = 4 - hmBitA * 4
        acc = acc + dlA * cfloat(qvalA - hmA) * sx[elemBaseA + tid]

        # Block B
        let blkStartB = rowBase + (blkIdx + 1) * 110
        let elemBaseB = (blkIdx + 1) * 256
        let dRawB = uint16(w[blkStartB + 108]) or (uint16(w[blkStartB + 109]) shl 8)
        let dAllB = hippoHalfToFloat(dRawB)
        let sByteB = cint(w[blkStartB + sByteOff])
        let tByteB = cint(w[blkStartB + tByteOff])
        let lowB = (sByteB shr useHighShift) and 0x0F
        let highB = ((tByteB shr auxShift) and 0x03) shl 4
        let scaleByteB = lowB or highB
        let scaleSignedB = ((scaleByteB xor 0x80) - 0x80)
        let dlB = dAllB * cfloat(scaleSignedB - 32)
        let qvalB = cint((w[blkStartB + 32 + qsByteIdx] shr shift) and 3)
        let hmBitB = (cint(w[blkStartB + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmB = 4 - hmBitB * 4
        acc = acc + dlB * cfloat(qvalB - hmB) * sx[elemBaseB + tid]

        # Block C
        let blkStartC = rowBase + (blkIdx + 2) * 110
        let elemBaseC = (blkIdx + 2) * 256
        let dRawC = uint16(w[blkStartC + 108]) or (uint16(w[blkStartC + 109]) shl 8)
        let dAllC = hippoHalfToFloat(dRawC)
        let sByteC = cint(w[blkStartC + sByteOff])
        let tByteC = cint(w[blkStartC + tByteOff])
        let lowC = (sByteC shr useHighShift) and 0x0F
        let highC = ((tByteC shr auxShift) and 0x03) shl 4
        let scaleByteC = lowC or highC
        let scaleSignedC = ((scaleByteC xor 0x80) - 0x80)
        let dlC = dAllC * cfloat(scaleSignedC - 32)
        let qvalC = cint((w[blkStartC + 32 + qsByteIdx] shr shift) and 3)
        let hmBitC = (cint(w[blkStartC + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmC = 4 - hmBitC * 4
        acc = acc + dlC * cfloat(qvalC - hmC) * sx[elemBaseC + tid]

        # Block D
        let blkStartD = rowBase + (blkIdx + 3) * 110
        let elemBaseD = (blkIdx + 3) * 256
        let dRawD = uint16(w[blkStartD + 108]) or (uint16(w[blkStartD + 109]) shl 8)
        let dAllD = hippoHalfToFloat(dRawD)
        let sByteD = cint(w[blkStartD + sByteOff])
        let tByteD = cint(w[blkStartD + tByteOff])
        let lowD = (sByteD shr useHighShift) and 0x0F
        let highD = ((tByteD shr auxShift) and 0x03) shl 4
        let scaleByteD = lowD or highD
        let scaleSignedD = ((scaleByteD xor 0x80) - 0x80)
        let dlD = dAllD * cfloat(scaleSignedD - 32)
        let qvalD = cint((w[blkStartD + 32 + qsByteIdx] shr shift) and 3)
        let hmBitD = (cint(w[blkStartD + hmaskByteOff]) shr hmaskBitPos) and 1
        let hmD = 4 - hmBitD * 4
        acc = acc + dlD * cfloat(qvalD - hmD) * sx[elemBaseD + tid]

        blkIdx = blkIdx + 4

      # Cleanup: remaining blocks
      while blkIdx < nBlocksPerRow:
        let blkStart = rowBase + blkIdx * 110
        let elemBase = blkIdx * 256
        let dRaw = uint16(w[blkStart + 108]) or (uint16(w[blkStart + 109]) shl 8)
        let dAll = hippoHalfToFloat(dRaw)
        let sByte = cint(w[blkStart + sByteOff])
        let tByte = cint(w[blkStart + tByteOff])
        let low = (sByte shr useHighShift) and 0x0F
        let high = ((tByte shr auxShift) and 0x03) shl 4
        let scaleByte = low or high
        let scaleSigned = ((scaleByte xor 0x80) - 0x80)
        let dl = dAll * cfloat(scaleSigned - 32)
        let qval = cint((w[blkStart + 32 + qsByteIdx] shr shift) and 3)
        let hmBit = (cint(w[blkStart + hmaskByteOff]) shr hmaskBitPos) and 1
        let hm = 4 - hmBit * 4
        acc = acc + dl * cfloat(qval - hm) * sx[elemBase + tid]
        blkIdx = blkIdx + 1

      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()

    reduceSum256(sdata, tid)

    if tid == 0 and outRow < int(outRows):
      outArr[outRow] = sdata[0]
    hippoSyncthreads()

proc gpuLinearColQ3K*(dst, x, wQuant: pointer, wCols, wRows: int,
                       stream: HippoStream) =
  if wCols > HippoMaxDecodeCols:
    raise newException(ValueError, "Q3K decode width exceeds limit: " & $wCols)
  let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
  let blk = newDim3(HippoBlockSize.uint32)
  var wPtr = wQuant; var xPtr = x; var dPtr = dst
  var outRowsArg = wRows.cint; var wColsArg = wCols.cint
  hippoLaunchKernel(linearQ3KDecodeKernel, gridDim = grid, blockDim = blk,
                    stream = stream,
                    args = hippoArgs(wPtr, xPtr, dPtr, outRowsArg, wColsArg))

proc gpuLinearColQuant*(dst, x, wQuant: pointer, wCols, wRows: int,
                         quantType: int32, stream: HippoStream) =
  ## Dispatch to the appropriate quantized GEMV kernel based on quant type.
  case quantType
  of GgmlTypeQ2K: gpuLinearColQ2K(dst, x, wQuant, wCols, wRows, stream)
  of GgmlTypeQ3K: gpuLinearColQ3K(dst, x, wQuant, wCols, wRows, stream)
  else: raise newException(ValueError, "unsupported quant type for GPU GEMV: " & $quantType)

proc ensureModelGpuPtrs*(m: var Model, hp: HParams) =
  ## Populate modelPtrs once by caching all weight device pointers.
  if modelPtrs.initialized:
    return
  ensureGpuContext()

  # Token embedding
  var tokEmb: Tensor
  try:
    tokEmb = m.getTensor("token_embd.weight")
  except KeyError:
    tokEmb = m.getTensor("tok_embeddings.weight")
  modelPtrs.tokEmb = cachedWeight("token_embd_or_tok_embeddings", tokEmb).devicePtr

  # Per-layer weights — use Q2_K raw upload when available, else float32
  modelPtrs.layers = newSeq[LayerGpuPtrs](hp.nLayer)
  for layer in 0 ..< hp.nLayer:
    let lp = "blk." & $layer & "."
    var lw: LayerGpuPtrs

    # Norms are always small float32 tensors
    lw.attnNorm = cachedWeight(lp & "attn_norm.weight", m.getTensor(lp & "attn_norm.weight")).devicePtr
    lw.ffnNorm = cachedWeight(lp & "ffn_norm.weight", m.getTensor(lp & "ffn_norm.weight")).devicePtr

    # Helper to upload a weight: quantized raw bytes or dequantized float32
    template uploadWeight(fp32Field, quantField, qtypeField: untyped, tensorSuffix: string) =
      let tn = lp & tensorSuffix
      let et = m.infos[tn].elemType.int32
      if et in {GgmlTypeQ2K.int32, GgmlTypeQ3K.int32}:
        let qw = cachedQuantWeight(tn, m, tn)
        quantField = qw.devicePtr
        fp32Field = nil
        qtypeField = et
      else:
        fp32Field = cachedWeight(tn, m.getTensor(tn)).devicePtr
        quantField = nil
        qtypeField = 0

    uploadWeight(lw.wq, lw.wqQ, lw.wqQType, "attn_q.weight")
    uploadWeight(lw.wk, lw.wkQ, lw.wkQType, "attn_k.weight")
    uploadWeight(lw.wv, lw.wvQ, lw.wvQType, "attn_v.weight")
    uploadWeight(lw.wo, lw.woQ, lw.woQType, "attn_output.weight")
    uploadWeight(lw.wGate, lw.wGateQ, lw.wGateQType, "ffn_gate.weight")
    uploadWeight(lw.wUp, lw.wUpQ, lw.wUpQType, "ffn_up.weight")
    uploadWeight(lw.wDown, lw.wDownQ, lw.wDownQType, "ffn_down.weight")

    # Store column counts needed for quant dispatch
    lw.wColsQ = hp.nEmb
    lw.wColsDown = hp.nFfn

    modelPtrs.layers[layer] = lw

  # Final norm + output
  var norm: Tensor
  try:
    norm = m.getTensor("output_norm.weight")
  except KeyError:
    norm = m.getTensor("norm.weight")
  modelPtrs.normWeight = cachedWeight("norm_or_output_norm.weight", norm).devicePtr

  # Output weight — handle both orientations
  let outW = m.getTensor("output.weight")
  let a0 = outW.shape[0]
  let a1 = outW.shape[1]
  if a0 == hp.nEmb and a1 == hp.nVocab:
    modelPtrs.outputWeight = cachedWeight("output.weight", outW).devicePtr
    modelPtrs.outputShape0 = a0
    modelPtrs.outputShape1 = a1
  elif a0 == hp.nVocab and a1 == hp.nEmb:
    let reshaped = outW.reshape(@[a1, a0])
    modelPtrs.outputWeight = cachedWeight("output.weight", reshaped).devicePtr
    modelPtrs.outputShape0 = a1
    modelPtrs.outputShape1 = a0
  else:
    raise newException(ValueError, "output weight shape mismatch")

  modelPtrs.initialized = true

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

  reduceSum256(sdata, tid)

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
  reduceMax256(sMax, tid)
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
  reduceSum256(sSum, tid)
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

  reduceMax256(sMax, tid)
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

  reduceSum256(sSum, tid)
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
# Kernel: Multi-row GEMV decode with shared input vector
# Opt-2 implementation: each block processes multiple rows
# ---------------------------------------------------------------------------
proc linearHippoDecodeKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  var sx {.hippoShared.}: array[HippoMaxDecodeCols, float32]
  var sdata {.hippoShared.}: array[HippoBlockSize, float32]
  let tid = int(threadIdx.x)
  let blockSize = int(blockDim.x)
  let cols = int(wCols)
  let unrollSpan = HippoDecodeDotUnroll * blockSize
  let baseRow = int(blockIdx.x) * HippoDecodeRowsPerBlock
  let wArray = cast[ptr UncheckedArray[float32]](wData)
  let xArray = cast[ptr UncheckedArray[float32]](xData)
  let outArray = cast[ptr UncheckedArray[float32]](outData)

  var k = tid
  while k < cols:
    sx[k] = xArray[k]
    k = k + blockSize
  hippoSyncthreads()

  for r in 0 ..< HippoDecodeRowsPerBlock:
    let outRow = baseRow + r

    if outRow < int(outRows):
      let rowBase = outRow * cols
      var acc = 0.0'f32
      k = tid
      while k + (HippoDecodeDotUnroll - 1) * blockSize < cols:
        let k1 = k + blockSize
        let k2 = k1 + blockSize
        let k3 = k2 + blockSize
        acc = acc + wArray[rowBase + k] * sx[k]
        acc = acc + wArray[rowBase + k1] * sx[k1]
        acc = acc + wArray[rowBase + k2] * sx[k2]
        acc = acc + wArray[rowBase + k3] * sx[k3]
        k = k + unrollSpan
      while k < cols:
        acc = acc + wArray[rowBase + k] * sx[k]
        k = k + blockSize
      sdata[tid] = acc
    else:
      sdata[tid] = 0.0'f32
    hippoSyncthreads()

    reduceSum256(sdata, tid)

    if tid == 0 and outRow < int(outRows):
      outArray[outRow] = sdata[0]
    hippoSyncthreads()

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
    if wCols > HippoMaxDecodeCols:
      raise newException(ValueError,
        "decode GEMV width exceeds HippoMaxDecodeCols: " & $wCols &
        " > " & $HippoMaxDecodeCols)
    # Optimized decode path: each block processes several output rows.
    let grid = newDim3(((wRows + HippoDecodeRowsPerBlock - 1) div HippoDecodeRowsPerBlock).uint32)
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
