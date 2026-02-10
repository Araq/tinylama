## Hippo-specific linear kernels and GPU caching for forward pass.

import
  std/tables,
  ./tensor

when not defined(cpp):
  {.error: "useHippo requires Nim's C++ backend. Build with `nim cpp`.".}

import hippo

const
  HippoBlockSizeX = 16
  HippoBlockSizeY = 16
  HippoDecodeBlockSize = 256

type
  HippoAllocRef = type(hippoMalloc(1))

  GpuWeightEntry = object
    bytes: int
    device: HippoAllocRef

  GpuLinearContext = object
    initialized: bool
    stream: HippoStream
    xBuf: HippoAllocRef
    outBuf: HippoAllocRef
    xCapBytes: int
    outCapBytes: int
    weights: Table[uint, GpuWeightEntry]

var gpuLinearContext: GpuLinearContext

proc tensorStorageKey(t: Tensor): uint =
  ## Create a cache key from the tensor storage base pointer.
  if t.data.len == 0:
    raise newException(ValueError, "cannot cache empty tensor storage")
  cast[uint](unsafeAddr t.data[0])

proc ensureGpuLinearContext() =
  ## Initialize the shared Hippo stream and caches once.
  if not gpuLinearContext.initialized:
    gpuLinearContext.stream = hippoStreamCreate()
    gpuLinearContext.weights = initTable[uint, GpuWeightEntry]()
    gpuLinearContext.initialized = true

proc ensureLinearBuffer(
  buf: var HippoAllocRef,
  capBytes: var int,
  wantedBytes: int
) =
  ## Grow the reusable device buffer when the requested size increases.
  if wantedBytes > capBytes:
    buf = hippoMalloc(wantedBytes)
    capBytes = wantedBytes

proc cachedWeightBuffer(w: Tensor, wBytes: int): pointer =
  ## Upload each weight tensor to device only once and reuse it.
  ensureGpuLinearContext()
  let
    key = tensorStorageKey(w)
    existing = gpuLinearContext.weights.getOrDefault(key)
  if existing.bytes == wBytes and existing.device != nil:
    return existing.device.p

  let devW = hippoMalloc(wBytes)
  hippoMemcpyAsync(
    devW.p,
    unsafeAddr w.data[0],
    wBytes,
    HippoMemcpyHostToDevice,
    gpuLinearContext.stream
  )
  gpuLinearContext.weights[key] = GpuWeightEntry(
    bytes: wBytes,
    device: devW
  )
  devW.p

proc linearHippoKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols, seqLen: cint
) {.hippoGlobal.} =
  ## Compute one output element for the ggml-column GEMM layout.
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

proc linearHippoDecodeKernel(
  wData, xData, outData: ptr float32,
  outRows, wCols: cint
) {.hippoGlobal.} =
  ## Compute one output row for the decode case where seqLen is one.
  let outRow = int(blockIdx.x * blockDim.x + threadIdx.x)
  if outRow < int(outRows):
    let wArray = cast[ptr UncheckedArray[float32]](wData)
    let xArray = cast[ptr UncheckedArray[float32]](xData)
    let outArray = cast[ptr UncheckedArray[float32]](outData)
    var acc = 0.0'f32
    for k in 0 ..< int(wCols):
      acc = acc + wArray[outRow * int(wCols) + k] * xArray[k]
    outArray[outRow] = acc

proc linearHippoCol*(x: Tensor, w: Tensor, wCols, wRows, seqLen: int): Tensor =
  ## Run ggml-column GEMM through Hippo on HIP/CUDA backends.
  result = newTensor(@[wRows, seqLen])
  if result.data.len == 0:
    return result

  ensureGpuLinearContext()

  let
    wBytes = w.data.len * sizeof(float32)
    xBytes = x.data.len * sizeof(float32)
    outBytes = result.data.len * sizeof(float32)
    stream = gpuLinearContext.stream

  ensureLinearBuffer(
    gpuLinearContext.xBuf,
    gpuLinearContext.xCapBytes,
    xBytes
  )
  ensureLinearBuffer(
    gpuLinearContext.outBuf,
    gpuLinearContext.outCapBytes,
    outBytes
  )

  let
    devW = cachedWeightBuffer(w, wBytes)
    devX = gpuLinearContext.xBuf.p
    devOut = gpuLinearContext.outBuf.p

  hippoMemcpyAsync(devX, unsafeAddr x.data[0], xBytes, HippoMemcpyHostToDevice, stream)

  var
    devWPtr = devW
    devXPtr = devX
    devOutPtr = devOut
    outRowsArg = wRows.cint
    wColsArg = wCols.cint

  if seqLen == 1:
    let
      blockDim = newDim3(HippoDecodeBlockSize.uint32, 1, 1)
      gridX = (wRows + HippoDecodeBlockSize - 1) div HippoDecodeBlockSize
      gridDim = newDim3(gridX.uint32, 1, 1)
    hippoLaunchKernel(
      linearHippoDecodeKernel,
      gridDim = gridDim,
      blockDim = blockDim,
      stream = stream,
      args = hippoArgs(devWPtr, devXPtr, devOutPtr, outRowsArg, wColsArg)
    )
  else:
    let
      blockDim = newDim3(HippoBlockSizeX.uint32, HippoBlockSizeY.uint32)
      gridX = (seqLen + HippoBlockSizeX - 1) div HippoBlockSizeX
      gridY = (wRows + HippoBlockSizeY - 1) div HippoBlockSizeY
      gridDim = newDim3(gridX.uint32, gridY.uint32)
    var seqLenArg = seqLen.cint
    hippoLaunchKernel(
      linearHippoKernel,
      gridDim = gridDim,
      blockDim = blockDim,
      stream = stream,
      args = hippoArgs(devWPtr, devXPtr, devOutPtr, outRowsArg, wColsArg, seqLenArg)
    )

  hippoMemcpyAsync(
    unsafeAddr result.data[0],
    devOut,
    outBytes,
    HippoMemcpyDeviceToHost,
    stream
  )
  hippoStreamSynchronize(stream)
