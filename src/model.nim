## Minimal model loader that reads GGUF tensors into float32.

import std/[tables]
when cpuEndian != littleEndian:
  import std/endians
import ./gguf_loader
import ./tensor
import ./quant

const
  GgmlTypeF32* = 0
  GgmlTypeF16* = 1
  GgmlTypeQ5_0* = 6
  GgmlTypeQ8_0* = 8
  GgmlTypeQ2K* = 10
  GgmlTypeQ3K* = 11
  GgmlTypeQ4K* = 12
  GgmlTypeQ6K* = 14

type
  HParams* = object
    arch*: string
    nVocab*: int
    nCtx*: int
    nEmb*: int
    nLayer*: int
    nFfn*: int
    nHead*: int
    nHeadKv*: int
    ropeDim*: int
    ropeFreqBase*: float32
    rmsEps*: float32

  Model* = object
    hparams*: HParams
    gguf*: GgufFile
    infos*: Table[string, GgufTensorInfo]
    cache*: Table[string, Tensor]

proc tensorElemCount*(info: GgufTensorInfo): int =
  var n = 1'u64
  for i in 0 ..< int(info.nDims):
    n *= info.ne[i]
  if n > uint64(high(int)):
    raise newException(ValueError, "tensor too large")
  int(n)

proc tensorShape(info: GgufTensorInfo): seq[int] =
  result = newSeq[int](int(info.nDims))
  for i in 0 ..< int(info.nDims):
    result[i] = int(info.ne[i])

proc loadTensorF32(g: GgufFile, info: GgufTensorInfo): Tensor =
  let count = tensorElemCount(info)
  result = newTensor(tensorShape(info))
  let dataPtr = tensorDataPtr(g, info)
  let rowLen = int(info.ne[0])
  let rows = if rowLen > 0: count div rowLen else: 0
  case info.elemType
  of GgmlTypeF32:
    copyMem(addr result.data[0], addr dataPtr[0], count * 4)
  of GgmlTypeF16:
    for i in 0 ..< count:
      var u = cast[ptr UncheckedArray[uint16]](addr dataPtr[i * 2])[0]
      when cpuEndian != littleEndian:
        u = swapEndian(u)
      result.data[i] = halfToFloat(u)
  of GgmlTypeQ2K:
    let rowSize = rowSizeQ2K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ2K(src, dst, rowLen)
  of GgmlTypeQ3K:
    let rowSize = rowSizeQ3K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ3K(src, dst, rowLen)
  of GgmlTypeQ5_0:
    let rowSize = rowSizeQ5_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ5_0(src, dst, rowLen)
  of GgmlTypeQ8_0:
    let rowSize = rowSizeQ8_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ8_0(src, dst, rowLen)
  of GgmlTypeQ4K:
    let rowSize = rowSizeQ4K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ4K(src, dst, rowLen)
  of GgmlTypeQ6K:
    let rowSize = rowSizeQ6K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr result.data[r * rowLen])
      dequantRowQ6K(src, dst, rowLen)
  else:
    raise newException(ValueError, "unsupported ggml type: " & $info.elemType)

proc getKvU32Arch(g: GgufFile, arch, suffix: string, outv: var int): bool =
  var v: uint32
  if arch.len > 0 and g.getKvU32(arch & "." & suffix, v):
    outv = int(v)
    return true
  if arch != "llama" and g.getKvU32("llama." & suffix, v):
    outv = int(v)
    return true
  false

proc getKvF32Arch(g: GgufFile, arch, suffix: string, outv: var float32): bool =
  if arch.len > 0 and g.getKvF32(arch & "." & suffix, outv):
    return true
  if arch != "llama" and g.getKvF32("llama." & suffix, outv):
    return true
  false

proc loadHParams(g: GgufFile): HParams =
  discard g.getKvStr("general.architecture", result.arch)
  let archPrefix = if result.arch.len > 0: result.arch else: "llama"
  discard getKvU32Arch(g, archPrefix, "vocab_size", result.nVocab)
  discard getKvU32Arch(g, archPrefix, "context_length", result.nCtx)
  discard getKvU32Arch(g, archPrefix, "embedding_length", result.nEmb)
  discard getKvU32Arch(g, archPrefix, "block_count", result.nLayer)
  discard getKvU32Arch(g, archPrefix, "feed_forward_length", result.nFfn)
  discard getKvU32Arch(g, archPrefix, "attention.head_count", result.nHead)
  discard getKvU32Arch(g, archPrefix, "attention.head_count_kv", result.nHeadKv)
  discard getKvU32Arch(g, archPrefix, "rope.dimension_count", result.ropeDim)
  discard getKvF32Arch(g, archPrefix, "rope.freq_base", result.ropeFreqBase)
  discard getKvF32Arch(g, archPrefix, "attention.layer_norm_rms_epsilon", result.rmsEps)
  if result.nHeadKv == 0:
    result.nHeadKv = result.nHead
  if result.nVocab == 0:
    var tokens: seq[string]
    if g.getKvArrStr("tokenizer.ggml.tokens", tokens):
      result.nVocab = tokens.len

proc loadModel*(path: string): Model =
  result.gguf = openGguf(path)
  result.hparams = loadHParams(result.gguf)
  result.infos = initTable[string, GgufTensorInfo](result.gguf.tensors.len * 2)
  result.cache = initTable[string, Tensor]()
  for info in result.gguf.tensors:
    result.infos[info.name] = info

proc close*(m: var Model) =
  m.gguf.close()

proc getTensor*(m: var Model, name: string): lent Tensor =
  if not m.cache.hasKey(name):
    if not m.infos.hasKey(name):
      raise newException(KeyError, "missing tensor: " & name)
    m.cache[name] = loadTensorF32(m.gguf, m.infos[name])
  m.cache[name]
