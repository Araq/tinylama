## Minimal model loader that reads GGUF tensors into float32.

import std/[tables]
when cpuEndian != littleEndian:
  import std/endians
import ./gguf_loader
import ./tensor
import ./quant

const
  GGML_TYPE_F32 = 0
  GGML_TYPE_F16 = 1
  GGML_TYPE_Q2_K = 10
  GGML_TYPE_Q3_K = 11
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
    rmsEps*: float32

  Model* = object
    hparams*: HParams
    gguf*: GgufFile
    infos*: Table[string, GgufTensorInfo]
    cache*: Table[string, Tensor]

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

proc loadTensorF32(g: GgufFile, info: GgufTensorInfo): Tensor =
  let count = tensorElemCount(info)
  result = newTensor(tensorShape(info))
  let dataPtr = tensorDataPtr(g, info)
  case info.elemType
  of GGML_TYPE_F32:
    copyMem(addr result.data[0], addr dataPtr[0], count * 4)
  of GGML_TYPE_F16:
    for i in 0 ..< count:
      var u = cast[ptr UncheckedArray[uint16]](addr dataPtr[i * 2])[0]
      when cpuEndian != littleEndian:
        u = swapEndian(u)
      result.data[i] = halfToFloat(u)
  of GGML_TYPE_Q2_K:
    dequantRowQ2K(dataPtr, cast[ptr UncheckedArray[float32]](addr result.data[0]), count)
  of GGML_TYPE_Q3_K:
    dequantRowQ3K(dataPtr, cast[ptr UncheckedArray[float32]](addr result.data[0]), count)
  of GGML_TYPE_Q6_K:
    dequantRowQ6K(dataPtr, cast[ptr UncheckedArray[float32]](addr result.data[0]), count)
  else:
    raise newException(ValueError, "unsupported ggml type: " & $info.elemType)

proc loadHParams(g: GgufFile): HParams =
  discard g.getKvStr("general.architecture", result.arch)
  var v: uint32
  if g.getKvU32("llama.vocab_size", v): result.nVocab = int(v)
  if g.getKvU32("llama.context_length", v): result.nCtx = int(v)
  if g.getKvU32("llama.embedding_length", v): result.nEmb = int(v)
  if g.getKvU32("llama.block_count", v): result.nLayer = int(v)
  if g.getKvU32("llama.feed_forward_length", v): result.nFfn = int(v)
  if g.getKvU32("llama.attention.head_count", v): result.nHead = int(v)
  if g.getKvU32("llama.attention.head_count_kv", v): result.nHeadKv = int(v)
  if g.getKvU32("llama.rope.dimension_count", v): result.ropeDim = int(v)
  var f: float32
  if g.getKvF32("llama.rope.freq_base", f): result.ropeFreqBase = f
  if g.getKvF32("llama.attention.layer_norm_rms_epsilon", f): result.rmsEps = f
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

proc getTensor*(m: var Model, name: string): Tensor =
  if m.cache.hasKey(name):
    return m.cache[name]
  if not m.infos.hasKey(name):
    raise newException(KeyError, "missing tensor: " & name)
  let t = loadTensorF32(m.gguf, m.infos[name])
  m.cache[name] = t
  t
