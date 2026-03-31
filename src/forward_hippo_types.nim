## Hippo GPU backend type definitions.
## Separated to break circular imports between forward_types and forward_hippo.

import hippo

type
  HippoAllocRef* = type(hippoMalloc(1))

  GpuTensor* = object
    devicePtr*: pointer       # raw device pointer
    alloc*: HippoAllocRef     # ref-counted allocation (prevents free)
    shape*: seq[int]
    sizeBytes*: int

  GpuKvCache* = object
    k*: seq[GpuTensor]        # [kvDim, maxLen] per layer
    v*: seq[GpuTensor]        # [kvDim, maxLen] per layer
    curLen*: int
    maxLen*: int
    nHeadKv*: int
    headDim*: int
