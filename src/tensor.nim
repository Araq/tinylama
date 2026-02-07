## Minimal float32 tensor utilities using Arraymancer.

import std/[math]
import arraymancer

type
  GGTensor* = object
    at*: arraymancer.Tensor[float32]

proc shape*(t: GGTensor): seq[int] =
  result = @[]
  for s in t.at.shape: result.add(s)
proc data*(t: GGTensor): seq[float32] = t.at.toSeq1D()

proc toGGTensor*(at: arraymancer.Tensor[float32]): GGTensor = GGTensor(at: at)

proc newGGTensor*(shape: seq[int]): GGTensor =
  result.at = newTensor[float32](shape)

proc reshape*(t: GGTensor, shape: seq[int]): GGTensor =
  result.at = t.at.reshape(shape)

proc transpose*(t: GGTensor): GGTensor =
  result.at = t.at.transpose()

proc add*(a, b: GGTensor): GGTensor =
  result.at = a.at + b.at

proc mul*(a, b: GGTensor): GGTensor =
  result.at = a.at * b.at

proc `/`*(a: GGTensor, b: float32): GGTensor =
  result.at = a.at / b

proc silu*(a: GGTensor): GGTensor =
  result.at = a.at.map(proc(x: float32): float32 = x / (1.0'f32 + exp(-x)))

proc gelu*(a: GGTensor): GGTensor =
  result.at = a.at.map(proc(x: float32): float32 = 0.5'f32 * x * (1.0'f32 + erf(x / sqrt(2.0'f32))))

proc matmul*(a, b: GGTensor): GGTensor =
  result.at = newTensor[float32](@[a.at.shape[0], b.at.shape[1]])
  gemm(1.0'f32, a.at, b.at, 0.0'f32, result.at)

proc softmax*(t: GGTensor): GGTensor =
  result.at = t.at.softmax()

proc rmsnorm*(x: GGTensor, weight: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[^1]
  result = newGGTensor(x.shape)
  for i in 0 ..< x.at.shape[0]:
    let row = x.at[i, _].reshape(dim)
    let ss = (row * row).sum()
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    result.at[i, _] = (row * inv * weight.at).reshape(1, dim)

proc rmsnormCols*(x: GGTensor, weight: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  result = newGGTensor(x.shape)
  for s in 0 ..< seqLen:
    let col = x.at[_, s].reshape(dim)
    let ss = (col * col).sum()
    let inv = 1.0'f32 / sqrt(ss / float32(dim) + eps)
    result.at[_, s] = (col * inv * weight.at).reshape(dim, 1)

proc layernormCols*(x: GGTensor, weight: GGTensor, bias: GGTensor, eps: float32): GGTensor =
  let dim = x.shape[0]
  let seqLen = x.shape[1]
  result = newGGTensor(x.shape)
  for s in 0 ..< seqLen:
    let col = x.at[_, s].reshape(dim)
    let mean = col.mean()
    let diff = col -. mean
    let variance = (diff * diff).mean()
    let inv = 1.0'f32 / sqrt(variance + eps)
    result.at[_, s] = ((diff * inv) * weight.at + bias.at).reshape(dim, 1)

proc ropeInplace*(x: var GGTensor, base: float32, startPos = 0) =
  # We should use a faster implementation later
  let dim = x.shape[^1]
  let seqLen = x.shape[^2]
  let outer = x.at.size div (seqLen * dim)
  let invBase = 1.0'f32 / base

  for o in 0 ..< outer:
    for p in 0 ..< seqLen:
      let pos = float32(startPos + p)
      for i in 0 ..< dim div 2:
        let theta = pow(invBase, float32(2 * i) / float32(dim))
        let angle = pos * theta
        let c = cos(angle)
        let s = sin(angle)
        let v0 = x.at[o, p, 2 * i]
        let v1 = x.at[o, p, 2 * i + 1]
        x.at[o, p, 2 * i] = v0 * c - v1 * s
        x.at[o, p, 2 * i + 1] = v0 * s + v1 * c
