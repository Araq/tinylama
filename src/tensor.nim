## Minimal float32 tensor utilities using Arraymancer.

import std/[math]
import arraymancer

when defined(cuda):
  type DeviceTensor* = CudaTensor[float32]
elif defined(opencl):
  type DeviceTensor* = ClTensor[float32]
else:
  type DeviceTensor* = arraymancer.Tensor[float32]

type
  GGTensor* = object
    at*: DeviceTensor

template toCpu*[T](t: arraymancer.Tensor[T]): arraymancer.Tensor[T] = t
when defined(cuda):
  template toCpu*[T](t: CudaTensor[T]): arraymancer.Tensor[T] = t.cpu()
when defined(opencl):
  template toCpu*[T](t: ClTensor[T]): arraymancer.Tensor[T] = t.cpu()

template toDevice*[T](t: arraymancer.Tensor[T]): DeviceTensor =
  when DeviceTensor is arraymancer.Tensor[float32]: t
  elif defined(cuda): t.cuda()
  elif defined(opencl): t.opencl()

proc shape*(t: GGTensor): seq[int] =
  result = @[]
  for s in t.at.shape: result.add(s)

proc toGGTensor*(at: DeviceTensor): GGTensor = GGTensor(at: at)

proc newGGTensor*(shape: seq[int]): GGTensor =
  let cpu = newTensor[float32](shape)
  result.at = cpu.toDevice()

proc reshape*(t: GGTensor, shape: seq[int]): GGTensor =
  result.at = t.at.reshape(shape)

proc transpose*(t: GGTensor): GGTensor =
  result.at = t.at.transpose()

proc add*(a, b: GGTensor): GGTensor =
  result.at = a.at +. b.at

proc mul*(a, b: GGTensor): GGTensor =
  result.at = a.at *. b.at

proc `/`*(a: GGTensor, b: float32): GGTensor =
  result.at = a.at /. b

proc silu*(a: GGTensor): GGTensor =
  when DeviceTensor is arraymancer.Tensor[float32]:
    result.at = a.at.map(proc(x: float32): float32 = x / (1.0'f32 + exp(-x)))
  else:
    let cpu = a.at.toCpu()
    let res = cpu.map(proc(x: float32): float32 = x / (1.0'f32 + exp(-x)))
    result.at = res.toDevice()

proc gelu*(a: GGTensor): GGTensor =
  when DeviceTensor is arraymancer.Tensor[float32]:
    result.at = a.at.map(proc(x: float32): float32 = 0.5'f32 * x * (1.0'f32 + erf(x / sqrt(2.0'f32))))
  else:
    let cpu = a.at.toCpu()
    let res = cpu.map(proc(x: float32): float32 = 0.5'f32 * x * (1.0'f32 + erf(x / sqrt(2.0'f32))))
    result.at = res.toDevice()

proc matmul*(a, b: GGTensor): GGTensor =
  # Robust matmul that handles transposed weights often found in GGUF
  let sA = a.at.shape
  let sB = b.at.shape
  if sA[^1] == sB[0]:
    result.at = a.at * b.at
  elif sA[0] == sB[0]:
    result.at = a.at.transpose() * b.at
  elif sA[^1] == sB[^1]:
    result.at = a.at * b.at.transpose()
  else:
    # Fallback to standard multiplication, which will likely throw a descriptive error if shapes are wrong
    result.at = a.at * b.at

proc softmax*(t: GGTensor): GGTensor =
  when DeviceTensor is arraymancer.Tensor[float32]:
    result.at = t.at.softmax()
  else:
    let cpu = t.at.toCpu()
    result.at = cpu.softmax().toDevice()

proc rmsnorm*(x: GGTensor, weight: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[^1]
  let cpuX = x.at.toCpu()
  let cpuW = weight.at.toCpu()
  var res = newTensor[float32](cpuX.shape)
  for i in 0 ..< cpuX.shape[0]:
    let row = cpuX[i, _].reshape(dim)
    let ss = (row *. row).sum()
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    res[i, _] = (row *. (inv *. cpuW)).reshape(1, dim)
  result.at = res.toDevice()

proc rmsnormCols*(x: GGTensor, weight: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  let cpuX = x.at.toCpu()
  let cpuW = weight.at.toCpu()
  var res = newTensor[float32](cpuX.shape)
  for s in 0 ..< seqLen:
    let col = cpuX[_, s].reshape(dim)
    let ss = (col *. col).sum()
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    res[_, s] = (col *. (inv *. cpuW)).reshape(dim, 1)
  result.at = res.toDevice()

proc layernormCols*(x: GGTensor, weight: GGTensor, bias: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  let cpuX = x.at.toCpu()
  let cpuW = weight.at.toCpu()
  let cpuB = bias.at.toCpu()
  var res = newTensor[float32](cpuX.shape)
  for s in 0 ..< seqLen:
    let col = cpuX[_, s].reshape(dim)
    let mean = col.mean()
    let diff = col -. mean
    let variance = (diff *. diff).mean()
    let inv = 1.0'f32 / sqrt(variance + eps)
    res[_, s] = ((diff *. inv) *. cpuW +. cpuB).reshape(dim, 1)
  result.at = res.toDevice()
