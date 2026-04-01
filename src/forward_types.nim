## Shared types for the forward pass, used by all backends.

import ./tensor

when defined(useHippo):
  import ./forward_hippo_types
  export forward_hippo_types

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
