## Shared types for the forward pass, used by all backends.

import ./[model, tensor]

when defined(useHippo):
  import ./forward_hippo

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
