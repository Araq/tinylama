## Minimal float32 tensor utilities (CPU, contiguous).

import std/[math]

type
  Tensor* = object
    data*: seq[float32]
    shape*: seq[int]
    strides*: seq[int]

proc computeStrides(shape: seq[int]): seq[int] =
  result = newSeq[int](shape.len)
  var stride = 1
  for i in countdown(shape.len - 1, 0):
    result[i] = stride
    stride *= shape[i]

proc numel*(shape: seq[int]): int =
  result = 1
  for s in shape:
    result *= s

proc newTensor*(shape: seq[int]): Tensor =
  result.shape = shape
  result.strides = computeStrides(shape)
  result.data = newSeq[float32](numel(shape))

proc reshape*(t: Tensor, shape: seq[int]): Tensor =
  if numel(shape) != t.data.len:
    raise newException(ValueError,
      "reshape: element count mismatch (" & $t.data.len & " vs " & $numel(shape) & ")")
  result.data = t.data
  result.shape = shape
  result.strides = computeStrides(shape)

proc checkSameShape(a, b: Tensor) =
  if a.shape != b.shape:
    raise newException(ValueError, "shape mismatch: " & $a.shape & " vs " & $b.shape)

proc add*(a, b: Tensor): Tensor =
  checkSameShape(a, b)
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len:
    result.data[i] = a.data[i] + b.data[i]

proc mul*(a, b: Tensor): Tensor =
  checkSameShape(a, b)
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len:
    result.data[i] = a.data[i] * b.data[i]

proc silu*(a: Tensor): Tensor =
  result = newTensor(a.shape)
  for i in 0 ..< a.data.len:
    let x = a.data[i]
    result.data[i] = x / (1.0'f32 + exp(-x))

proc matmul*(a, b: Tensor): Tensor =
  ## 2D matmul: (m x k) * (k x n) -> (m x n)
  if a.shape.len != 2 or b.shape.len != 2:
    raise newException(ValueError, "matmul: expects 2D tensors")
  let m = a.shape[0]
  let k = a.shape[1]
  let kb = b.shape[0]
  let n = b.shape[1]
  if k != kb:
    raise newException(ValueError, "matmul: inner dim mismatch")
  result = newTensor(@[m, n])
  for i in 0 ..< m:
    let arow = i * k
    let crow = i * n
    for j in 0 ..< n:
      var acc = 0.0'f32
      for p in 0 ..< k:
        acc += a.data[arow + p] * b.data[p * n + j]
      result.data[crow + j] = acc

proc rmsnorm*(x: Tensor, weight: Tensor, eps: float32): Tensor =
  ## Normalize last dimension and apply per-dim weight.
  if x.shape.len < 1:
    raise newException(ValueError, "rmsnorm: empty shape")
  if weight.shape.len != 1 or weight.shape[0] != x.shape[^1]:
    raise newException(ValueError, "rmsnorm: weight shape mismatch")
  let dim = x.shape[^1]
  let outer = x.data.len div dim
  result = newTensor(x.shape)
  for o in 0 ..< outer:
    var ss = 0.0'f32
    let base = o * dim
    for i in 0 ..< dim:
      let v = x.data[base + i]
      ss += v * v
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    for i in 0 ..< dim:
      result.data[base + i] = x.data[base + i] * inv * weight.data[i]

proc rmsnormCols*(x: Tensor, weight: Tensor, eps: float32): Tensor =
  ## Normalize over rows for each column (ggml column layout).
  if x.shape.len != 2:
    raise newException(ValueError, "rmsnormCols: expects 2D tensor")
  if weight.shape.len != 1 or weight.shape[0] != x.shape[0]:
    raise newException(ValueError, "rmsnormCols: weight shape mismatch")
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  result = newTensor(x.shape)
  for s in 0 ..< seqLen:
    var ss = 0.0'f32
    for r in 0 ..< dim:
      let v = x.data[r * seqLen + s]
      ss += v * v
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    for r in 0 ..< dim:
      result.data[r * seqLen + s] = x.data[r * seqLen + s] * inv * weight.data[r]

proc softmaxLastDim*(x: Tensor): Tensor =
  if x.shape.len < 1:
    raise newException(ValueError, "softmax: empty shape")
  let dim = x.shape[^1]
  let outer = x.data.len div dim
  result = newTensor(x.shape)
  for o in 0 ..< outer:
    let base = o * dim
    var maxv = x.data[base]
    for i in 1 ..< dim:
      maxv = max(maxv, x.data[base + i])
    var sum = 0.0'f32
    for i in 0 ..< dim:
      let v = exp(x.data[base + i] - maxv)
      result.data[base + i] = v
      sum += v
    let inv = 1.0'f32 / sum
    for i in 0 ..< dim:
      result.data[base + i] *= inv

proc ropeInplace*(x: var Tensor, base: float32, startPos = 0) =
  ## Applies RoPE in-place on the last dimension (must be even).
  if x.shape.len < 2:
    raise newException(ValueError, "rope: expected at least 2D tensor")
  let dim = x.shape[^1]
  if (dim mod 2) != 0:
    raise newException(ValueError, "rope: last dim must be even")
  let seqLen = x.shape[^2]
  let outer = x.data.len div (seqLen * dim)
  let invBase = 1.0'f32 / base
  for o in 0 ..< outer:
    let outerBase = o * seqLen * dim
    for p in 0 ..< seqLen:
      let pos = float32(startPos + p)
      let row = outerBase + p * dim
      for i in 0 ..< dim div 2:
        let idx0 = row + 2 * i
        let idx1 = row + 2 * i + 1
        let theta = pow(invBase, float32(2 * i) / float32(dim))
        let angle = pos * theta
        let c = cos(angle)
        let s = sin(angle)
        let v0 = x.data[idx0]
        let v1 = x.data[idx1]
        x.data[idx0] = v0 * c - v1 * s
        x.data[idx1] = v0 * s + v1 * c
