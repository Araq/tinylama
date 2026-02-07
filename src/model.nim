## Minimal model loader that reads GGUF tensors into float32.

import std/[tables]
when cpuEndian != littleEndian:
  import std/endians
import arraymancer
import ./gguf_loader
import ./tensor
import ./quant

const
  GGML_TYPE_F32 = 0
  GGML_TYPE_F16 = 1
  GGML_TYPE_Q4_0 = 2
  GGML_TYPE_Q8_0 = 7
  GGML_TYPE_Q2_K = 10
  GGML_TYPE_Q3_K = 11
  GGML_TYPE_Q4_K = 12
  GGML_TYPE_Q5_K = 13
  GGML_TYPE_Q6_K = 14

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
    ropeFreqScale*: float32
    rmsEps*: float32
    actType*: string
    normType*: string
    ropeType*: string

  Model* = object
    hparams*: HParams
    gguf*: GgufFile
    infos*: Table[string, GgufTensorInfo]
    cache*: Table[string, GGTensor]

proc tensorElemCount(info: GgufTensorInfo): int =
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

proc loadTensorF32(g: GgufFile, info: GgufTensorInfo): GGTensor =
  let count = tensorElemCount(info)
  result = newGGTensor(tensorShape(info))
  let dataPtr = tensorDataPtr(g, info)
  let rowLen = int(info.ne[0])
  let rows = if rowLen > 0: count div rowLen else: 0
  let dstPtr = cast[ptr UncheckedArray[float32]](arraymancer.get_data_ptr(result.at))
  case info.elemType
  of GGML_TYPE_F32:
    copyMem(dstPtr, dataPtr, count * 4)
  of GGML_TYPE_F16:
    for i in 0 ..< count:
      var u = cast[ptr UncheckedArray[uint16]](addr dataPtr[i * 2])[0]
      when cpuEndian != littleEndian:
        u = swapEndian(u)
      dstPtr[i] = halfToFloat(u)
  of GGML_TYPE_Q4_0:
    let rowSize = rowSizeQ4_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ4_0(src, dst, rowLen)
  of GGML_TYPE_Q8_0:
    let rowSize = rowSizeQ8_0(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ8_0(src, dst, rowLen)
  of GGML_TYPE_Q2_K:
    let rowSize = rowSizeQ2K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ2K(src, dst, rowLen)
  of GGML_TYPE_Q3_K:
    let rowSize = rowSizeQ3K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ3K(src, dst, rowLen)
  of GGML_TYPE_Q4_K:
    let rowSize = rowSizeQ4K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ4K(src, dst, rowLen)
  of GGML_TYPE_Q5_K:
    let rowSize = rowSizeQ5K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ5K(src, dst, rowLen)
  of GGML_TYPE_Q6_K:
    let rowSize = rowSizeQ6K(rowLen)
    for r in 0 ..< rows:
      let src = cast[ptr UncheckedArray[byte]](addr dataPtr[r * rowSize])
      let dst = cast[ptr UncheckedArray[float32]](addr dstPtr[r * rowLen])
      dequantRowQ6K(src, dst, rowLen)
  else:
    raise newException(ValueError, "unsupported ggml type: " & $info.elemType)

proc loadHParams(g: GgufFile): HParams =
  discard g.getKvStr("general.architecture", result.arch)
  if result.arch == "": result.arch = "llama"
  let p = result.arch & "."
  var v: uint32
  if g.getKvU32(p & "vocab_size", v) or g.getKvU32("llama.vocab_size", v): result.nVocab = int(v)
  if g.getKvU32(p & "context_length", v) or g.getKvU32("llama.context_length", v): result.nCtx = int(v)
  if g.getKvU32(p & "embedding_length", v) or g.getKvU32("llama.embedding_length", v): result.nEmb = int(v)
  if g.getKvU32(p & "block_count", v) or g.getKvU32("llama.block_count", v): result.nLayer = int(v)
  if g.getKvU32(p & "feed_forward_length", v) or g.getKvU32("llama.feed_forward_length", v): result.nFfn = int(v)
  if g.getKvU32(p & "attention.head_count", v) or g.getKvU32("llama.attention.head_count", v): result.nHead = int(v)
  if g.getKvU32(p & "attention.head_count_kv", v) or g.getKvU32("llama.attention.head_count_kv", v): result.nHeadKv = int(v)
  if g.getKvU32(p & "rope.dimension_count", v) or g.getKvU32("llama.rope.dimension_count", v): result.ropeDim = int(v)
  var f: float32
  if g.getKvF32(p & "rope.freq_base", f) or g.getKvF32("llama.rope.freq_base", f): result.ropeFreqBase = f
  if g.getKvF32(p & "rope.freq_scale", f): result.ropeFreqScale = f
  if g.getKvF32(p & "attention.layer_norm_rms_epsilon", f) or g.getKvF32("llama.attention.layer_norm_rms_epsilon", f) or g.getKvF32(p & "attention.layer_norm_epsilon", f): result.rmsEps = f

  if result.ropeFreqBase == 0: result.ropeFreqBase = 10000.0
  if result.nHeadKv == 0: result.nHeadKv = result.nHead

  # Architecture specific defaults and overrides
  result.actType = "silu"
  result.normType = "rms"
  result.ropeType = "llama"

  if result.arch == "gemma":
    result.actType = "gelu"
    result.normType = "rms"
  elif result.arch == "phi3":
    result.actType = "silu"
    result.normType = "rms"
  elif result.arch == "qwen2":
    result.actType = "silu"
    result.normType = "rms"

  if result.nVocab == 0:
    var tokens: seq[string]
    if g.getKvArrStr("tokenizer.ggml.tokens", tokens):
      result.nVocab = tokens.len

proc loadModel*(path: string): Model =
  result.gguf = openGguf(path)
  result.hparams = loadHParams(result.gguf)
  result.infos = initTable[string, GgufTensorInfo](result.gguf.tensors.len * 2)
  result.cache = initTable[string, GGTensor]()
  for info in result.gguf.tensors:
    result.infos[info.name] = info

proc close*(m: var Model) =
  m.gguf.close()

proc getTensor*(m: var Model, name: string): GGTensor =
  if m.cache.hasKey(name):
    return m.cache[name]
  if not m.infos.hasKey(name):
    raise newException(KeyError, "missing tensor: " & name)
  let t = loadTensorF32(m.gguf, m.infos[name])
  m.cache[name] = t
  t
