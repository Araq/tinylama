## Minimal GGUF loader with memfiles for demand paging.
## Focused on reading metadata, KV pairs, and tensor table.

import std/[memfiles]

when cpuEndian != littleEndian:
  import std/endians

const
  ggufMagic* = "GGUF"
  ggufVersion* = 3'u32
  ggufDefaultAlignment* = 32'u32
  ggufMaxDims* = 4

type
  GgufType* = enum
    ggufUint8 = 0,
    ggufInt8 = 1,
    ggufUint16 = 2,
    ggufInt16 = 3,
    ggufUint32 = 4,
    ggufInt32 = 5,
    ggufFloat32 = 6,
    ggufBool = 7,
    ggufString = 8,
    ggufArray = 9,
    ggufUint64 = 10,
    ggufInt64 = 11,
    ggufFloat64 = 12

  GgufArray* = object
    elemType*: GgufType
    n*: uint64
    data*: seq[byte]
    strs*: seq[string]

  GgufValueKind* = enum
    gvkU8, gvkI8, gvkU16, gvkI16, gvkU32, gvkI32,
    gvkF32, gvkBool, gvkStr, gvkArr, gvkU64, gvkI64, gvkF64

  GgufValue* = object
    case kind*: GgufValueKind
    of gvkU8:  u8*: uint8
    of gvkI8:  i8*: int8
    of gvkU16: u16*: uint16
    of gvkI16: i16*: int16
    of gvkU32: u32*: uint32
    of gvkI32: i32*: int32
    of gvkF32: f32*: float32
    of gvkBool: b*: bool
    of gvkStr: s*: string
    of gvkArr: arr*: GgufArray
    of gvkU64: u64*: uint64
    of gvkI64: i64*: int64
    of gvkF64: f64*: float64

  GgufKv* = object
    key*: string
    value*: GgufValue

  GgufTensorInfo* = object
    name*: string
    nDims*: uint32
    ne*: array[ggufMaxDims, uint64]
    elemType*: int32  ## raw ggml_type id
    offset*: uint64   ## relative to start of data section

  GgufFile* = object
    mem*: MemFile
    data*: ptr UncheckedArray[byte]
    size*: int
    alignment*: uint32
    dataOffset*: uint64
    kv*: seq[GgufKv]
    tensors*: seq[GgufTensorInfo]

when cpuEndian == littleEndian:
  template fromLE*(x: untyped): untyped = x
else:
  proc fromLE*(x: uint16): uint16 =
    swapEndian16(addr result, addr x)
  proc fromLE*(x: uint32): uint32 =
    swapEndian32(addr result, addr x)
  proc fromLE*(x: uint64): uint64 =
    swapEndian64(addr result, addr x)
  proc fromLE*(x: int16): int16 =
    swapEndian16(addr result, addr x)
  proc fromLE*(x: int32): int32 =
    swapEndian32(addr result, addr x)
  proc fromLE*(x: int64): int64 =
    swapEndian64(addr result, addr x)

proc close*(g: var GgufFile) =
  memfiles.close(g.mem)
  g.data = nil
  g.size = 0

template ensure(cond: bool, msg: string) =
  if not cond:
    raise newException(IOError, msg)

proc readAt[T](data: ptr UncheckedArray[byte], size: int, pos: var int): T =
  ensure(pos + sizeof(T) <= size, "GGUF: truncated file")
  var tmp: T
  copyMem(addr tmp, addr data[pos], sizeof(T))
  pos += sizeof(T)
  tmp

proc readU8(data: ptr UncheckedArray[byte], size: int, pos: var int): uint8 =
  readAt[uint8](data, size, pos)

proc readI8(data: ptr UncheckedArray[byte], size: int, pos: var int): int8 =
  readAt[int8](data, size, pos)

proc readU16(data: ptr UncheckedArray[byte], size: int, pos: var int): uint16 =
  fromLE(readAt[uint16](data, size, pos))

proc readI16(data: ptr UncheckedArray[byte], size: int, pos: var int): int16 =
  fromLE(readAt[int16](data, size, pos))

proc readU32(data: ptr UncheckedArray[byte], size: int, pos: var int): uint32 =
  fromLE(readAt[uint32](data, size, pos))

proc readI32(data: ptr UncheckedArray[byte], size: int, pos: var int): int32 =
  fromLE(readAt[int32](data, size, pos))

proc readU64(data: ptr UncheckedArray[byte], size: int, pos: var int): uint64 =
  fromLE(readAt[uint64](data, size, pos))

proc readI64(data: ptr UncheckedArray[byte], size: int, pos: var int): int64 =
  fromLE(readAt[int64](data, size, pos))

proc readF32(data: ptr UncheckedArray[byte], size: int, pos: var int): float32 =
  cast[float32](readU32(data, size, pos))

proc readF64(data: ptr UncheckedArray[byte], size: int, pos: var int): float64 =
  cast[float64](readU64(data, size, pos))

proc readBool(data: ptr UncheckedArray[byte], size: int, pos: var int): bool =
  readAt[int8](data, size, pos) != 0

proc readString(data: ptr UncheckedArray[byte], size: int, pos: var int): string =
  let n = readU64(data, size, pos)
  if n > uint64(high(int)):
    raise newException(IOError, "GGUF: string length too large at " & $pos)
  if pos + int(n) > size:
    raise newException(IOError, "GGUF: truncated string at " & $pos &
      " len=" & $n & " size=" & $size)
  result = newString(int(n))
  if n > 0:
    copyMem(addr result[0], addr data[pos], int(n))
  pos += int(n)

proc readArray(data: ptr UncheckedArray[byte], size: int, pos: var int): GgufArray =
  let elemType = GgufType(readU32(data, size, pos))
  let n = readU64(data, size, pos)
  result.elemType = elemType
  result.n = n
  if elemType == ggufString:
    result.strs.setLen(int(n))
    for i in 0 ..< int(n):
      result.strs[i] = readString(data, size, pos)
  else:
    let elemSize = case elemType
      of ggufUint8, ggufInt8, ggufBool: 1
      of ggufUint16, ggufInt16: 2
      of ggufUint32, ggufInt32, ggufFloat32: 4
      of ggufUint64, ggufInt64, ggufFloat64: 8
      of ggufArray, ggufString:
        ensure(false, "GGUF: invalid array elem type")
        0
    let total = int(n) * elemSize
    if pos + total > size:
      raise newException(IOError, "GGUF: truncated array at " & $pos &
        " bytes=" & $total & " size=" & $size)
    result.data = newSeq[byte](total)
    copyMem(addr result.data[0], addr data[pos], total)
    pos += total

proc readValue(data: ptr UncheckedArray[byte], size: int, pos: var int, typ: GgufType): GgufValue =
  case typ
  of ggufUint8:   GgufValue(kind: gvkU8,  u8: readU8(data, size, pos))
  of ggufInt8:    GgufValue(kind: gvkI8,  i8: readI8(data, size, pos))
  of ggufUint16:  GgufValue(kind: gvkU16, u16: readU16(data, size, pos))
  of ggufInt16:   GgufValue(kind: gvkI16, i16: readI16(data, size, pos))
  of ggufUint32:  GgufValue(kind: gvkU32, u32: readU32(data, size, pos))
  of ggufInt32:   GgufValue(kind: gvkI32, i32: readI32(data, size, pos))
  of ggufFloat32: GgufValue(kind: gvkF32, f32: readF32(data, size, pos))
  of ggufBool:    GgufValue(kind: gvkBool, b: readBool(data, size, pos))
  of ggufString:  GgufValue(kind: gvkStr, s: readString(data, size, pos))
  of ggufArray:   GgufValue(kind: gvkArr, arr: readArray(data, size, pos))
  of ggufUint64:  GgufValue(kind: gvkU64, u64: readU64(data, size, pos))
  of ggufInt64:   GgufValue(kind: gvkI64, i64: readI64(data, size, pos))
  of ggufFloat64: GgufValue(kind: gvkF64, f64: readF64(data, size, pos))

proc alignUp(x: uint64, alignment: uint32): uint64 =
  let a = uint64(alignment)
  (x + a - 1) and not (a - 1)

proc openGguf*(path: string; debug = false): GgufFile =
  result.mem = memfiles.open(path, fmRead)
  result.data = cast[ptr UncheckedArray[byte]](result.mem.mem)
  result.size = result.mem.size
  result.alignment = ggufDefaultAlignment
  var pos = 0

  ensure(result.size >= 4, "GGUF: file too small")
  var magic: array[4, byte]
  copyMem(addr magic[0], addr result.data[0], 4)
  ensure(magic[0] == byte('G') and magic[1] == byte('G') and
         magic[2] == byte('U') and magic[3] == byte('F'),
         "GGUF: invalid magic")
  pos = 4

  let version = readU32(result.data, result.size, pos)
  ensure(version >= 2'u32, "GGUF: version 1 is not supported")
  let nTensors = readU64(result.data, result.size, pos)
  let nKv = readU64(result.data, result.size, pos)

  result.kv.setLen(int(nKv))
  for i in 0 ..< int(nKv):
    let key = readString(result.data, result.size, pos)
    let typ = GgufType(readU32(result.data, result.size, pos))
    let value = readValue(result.data, result.size, pos, typ)
    result.kv[i] = GgufKv(key: key, value: value)
    if key == "general.alignment" and value.kind == gvkU32:
      result.alignment = value.u32

  result.tensors.setLen(int(nTensors))
  for i in 0 ..< int(nTensors):
    var info: GgufTensorInfo
    info.name = readString(result.data, result.size, pos)
    info.nDims = readU32(result.data, result.size, pos)
    ensure(info.nDims <= ggufMaxDims.uint32, "GGUF: invalid tensor dims")
    for d in 0 ..< ggufMaxDims:
      info.ne[d] = 1'u64
    for d in 0 ..< int(info.nDims):
      info.ne[d] = readU64(result.data, result.size, pos)
    info.elemType = readI32(result.data, result.size, pos)
    info.offset = readU64(result.data, result.size, pos)
    result.tensors[i] = info

  result.dataOffset = alignUp(uint64(pos), result.alignment)

proc findKv*(g: GgufFile, key: string, value: var GgufValue): bool =
  for kv in g.kv:
    if kv.key == key:
      value = kv.value
      return true
  false

proc getKvU32*(g: GgufFile, key: string, value: var uint32): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkU32:
    value = v.u32
    return true
  false

proc getKvI32*(g: GgufFile, key: string, value: var int32): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkI32:
    value = v.i32
    return true
  false

proc getKvBool*(g: GgufFile, key: string, value: var bool): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkBool:
    value = v.b
    return true
  false

proc getKvF32*(g: GgufFile, key: string, value: var float32): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkF32:
    value = v.f32
    return true
  false

proc getKvStr*(g: GgufFile, key: string, value: var string): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkStr:
    value = v.s
    return true
  false

proc getKvArrStr*(g: GgufFile, key: string, value: var seq[string]): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkArr and v.arr.elemType == ggufString:
    value = v.arr.strs
    return true
  false

proc getKvArrF32*(g: GgufFile, key: string, value: var seq[float32]): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkArr and v.arr.elemType == ggufFloat32:
    let n = int(v.arr.n)
    value.setLen(n)
    for i in 0 ..< n:
      let offset = i * 4
      var tmp: uint32
      copyMem(addr tmp, addr v.arr.data[offset], 4)
      value[i] = cast[float32](fromLE(tmp))
    return true
  false

proc getKvArrI32*(g: GgufFile, key: string, value: var seq[int32]): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkArr and v.arr.elemType == ggufInt32:
    let n = int(v.arr.n)
    value.setLen(n)
    for i in 0 ..< n:
      let offset = i * 4
      var tmp: int32
      copyMem(addr tmp, addr v.arr.data[offset], 4)
      value[i] = fromLE(tmp)
    return true
  false

proc getKvArrU32*(g: GgufFile, key: string, value: var seq[uint32]): bool =
  var v: GgufValue
  if g.findKv(key, v) and v.kind == gvkArr and v.arr.elemType == ggufUint32:
    let n = int(v.arr.n)
    value.setLen(n)
    for i in 0 ..< n:
      let offset = i * 4
      var tmp: uint32
      copyMem(addr tmp, addr v.arr.data[offset], 4)
      value[i] = fromLE(tmp)
    return true
  false

proc tensorDataPtr*(g: GgufFile, info: GgufTensorInfo): ptr UncheckedArray[byte] =
  let off = int(g.dataOffset + info.offset)
  if off < 0 or off >= g.size:
    raise newException(IOError, "GGUF: tensor data offset out of range")
  cast[ptr UncheckedArray[byte]](addr g.data[off])
