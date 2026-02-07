## Dequantization helpers for GGUF tensors.

import std/[math]
when cpuEndian != littleEndian:
  import std/endians

const
  QK_K* = 256
  blockQ4_0Size* = 2 + 16
  blockQ4_1Size* = 2 + 2 + 16
  blockQ5_0Size* = 2 + 4 + 16
  blockQ5_1Size* = 2 + 4 + 4 + 16
  blockQ8_0Size* = 2 + 32
  blockQ2KSize* = 2 + 2 + (QK_K div 16) + (QK_K div 4) # d, dmin, scales, qs
  blockQ3KSize* = 2 + (QK_K div 4) + (QK_K div 8) + 12  # d, qs, hmask, scales
  blockQ4KSize* = 2 + 2 + 12 + (QK_K div 2) # d, dmin, scales, qs
  blockQ5KSize* = 2 + 2 + 12 + (QK_K div 2) + (QK_K div 8) # d, dmin, scales, qs, qh
  blockQ6KSize* = 2 + (QK_K div 16) + 3 * (QK_K div 4) # d, scales, ql+qh

proc rowSizeQ4_0*(rowLen: int): int = (rowLen div 32) * blockQ4_0Size
proc rowSizeQ4_1*(rowLen: int): int = (rowLen div 32) * blockQ4_1Size
proc rowSizeQ5_0*(rowLen: int): int = (rowLen div 32) * blockQ5_0Size
proc rowSizeQ5_1*(rowLen: int): int = (rowLen div 32) * blockQ5_1Size
proc rowSizeQ8_0*(rowLen: int): int = (rowLen div 32) * blockQ8_0Size

proc rowSizeQ2K*(rowLen: int): int = (rowLen div QK_K) * blockQ2KSize
proc rowSizeQ3K*(rowLen: int): int = (rowLen div QK_K) * blockQ3KSize
proc rowSizeQ4K*(rowLen: int): int = (rowLen div QK_K) * blockQ4KSize
proc rowSizeQ5K*(rowLen: int): int = (rowLen div QK_K) * blockQ5KSize
proc rowSizeQ6K*(rowLen: int): int = (rowLen div QK_K) * blockQ6KSize

proc halfToFloat*(h: uint16): float32 =
  let s = (h shr 15) and 0x1
  let e = (h shr 10) and 0x1F
  let f = h and 0x3FF
  if e == 0:
    if f == 0: return (if s == 1: -0.0'f32 else: 0.0'f32)
    return (if s == 1: -1.0'f32 else: 1.0'f32) * pow(2.0'f32, -14.0'f32) * (float32(f) / 1024.0'f32)
  elif e == 31:
    if f == 0: return (if s == 1: -Inf else: Inf)
    return NaN
  else:
    let exp = int(e) - 15
    let mant = 1.0'f32 + float32(f) / 1024.0'f32
    return (if s == 1: -1.0'f32 else: 1.0'f32) * mant * pow(2.0'f32, float32(exp))

proc dequantRowQ4_0*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  for b in 0 ..< k div 32:
    let off = b * blockQ4_0Size
    var dRaw: uint16
    copyMem(addr dRaw, addr src[off], 2)
    when cpuEndian != littleEndian: dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)
    for i in 0 ..< 16:
      let q = src[off + 2 + i]
      dst[b*32 + i] = d * float32(int8(q and 0xF) - 8)
      dst[b*32 + i + 16] = d * float32(int8(q shr 4) - 8)

proc dequantRowQ4_1*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  for b in 0 ..< k div 32:
    let off = b * blockQ4_1Size
    var dRaw, mRaw: uint16
    copyMem(addr dRaw, addr src[off], 2)
    copyMem(addr mRaw, addr src[off + 2], 2)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      mRaw = swapEndian(mRaw)
    let d = halfToFloat(dRaw)
    let m = halfToFloat(mRaw)
    for i in 0 ..< 16:
      let q = src[off + 4 + i]
      dst[b*32 + i] = d * float32(q and 0xF) + m
      dst[b*32 + i + 16] = d * float32(q shr 4) + m

proc dequantRowQ5_0*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  for b in 0 ..< k div 32:
    let off = b * blockQ5_0Size
    var dRaw: uint16
    copyMem(addr dRaw, addr src[off], 2)
    var qh: uint32
    copyMem(addr qh, addr src[off + 2], 4)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      qh = swapEndian(qh)
    let d = halfToFloat(dRaw)
    for i in 0 ..< 16:
      let q = src[off + 6 + i]
      let h1 = if (qh and (1'u32 shl i)) != 0: 16'i8 else: 0'i8
      let h2 = if (qh and (1'u32 shl (i + 16))) != 0: 16'i8 else: 0'i8
      dst[b*32 + i] = d * float32(int8(q and 0xF) + h1 - 16)
      dst[b*32 + i + 16] = d * float32(int8(q shr 4) + h2 - 16)

proc dequantRowQ5_1*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  for b in 0 ..< k div 32:
    let off = b * blockQ5_1Size
    var dRaw, mRaw: uint16
    copyMem(addr dRaw, addr src[off], 2)
    copyMem(addr mRaw, addr src[off + 2], 2)
    var qh: uint32
    copyMem(addr qh, addr src[off + 4], 4)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      mRaw = swapEndian(mRaw)
      qh = swapEndian(qh)
    let d = halfToFloat(dRaw)
    let m = halfToFloat(mRaw)
    for i in 0 ..< 16:
      let q = src[off + 8 + i]
      let h1 = if (qh and (1'u32 shl i)) != 0: 16'u8 else: 0'u8
      let h2 = if (qh and (1'u32 shl (i + 16))) != 0: 16'u8 else: 0'u8
      dst[b*32 + i] = d * float32((q and 0xF) or h1) + m
      dst[b*32 + i + 16] = d * float32((q shr 4) or h2) + m

proc dequantRowQ8_0*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  for b in 0 ..< k div 32:
    let off = b * blockQ8_0Size
    var dRaw: uint16
    copyMem(addr dRaw, addr src[off], 2)
    when cpuEndian != littleEndian: dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)
    for i in 0 ..< 32:
      dst[b*32 + i] = d * float32(cast[ptr UncheckedArray[int8]](addr src[off + 2])[i])

proc dequantRowQ2K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let scalesPtr = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qsPtr = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 16)])
    var dRaw, dminRaw: uint16
    copyMem(addr dRaw, addr src[offset + (QK_K div 16) + (QK_K div 4)], 2)
    copyMem(addr dminRaw, addr src[offset + (QK_K div 16) + (QK_K div 4) + 2], 2)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    var scaleIdx = 0
    var qOffset = 0
    for n in 0 ..< 2:
      var shift = 0
      for _ in 0 ..< 4:
        var sc = scalesPtr[scaleIdx]
        inc scaleIdx
        var dl = d * float32(sc and 0x0F)
        var ml = dmin * float32(sc shr 4)
        for l in 0 ..< 16:
          dst[outIdx] = dl * float32((qsPtr[qOffset + l] shr shift) and 3) - ml
          inc outIdx
        sc = scalesPtr[scaleIdx]
        inc scaleIdx
        dl = d * float32(sc and 0x0F)
        ml = dmin * float32(sc shr 4)
        for l in 0 ..< 16:
          dst[outIdx] = dl * float32((qsPtr[qOffset + 16 + l] shr shift) and 3) - ml
          inc outIdx
        shift += 2
      qOffset += 32
    offset += blockQ2KSize

proc dequantRowQ3K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let hmask = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qs = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 8)])
    var dRaw: uint16
    copyMem(addr dRaw, addr src[offset + (QK_K div 8) + (QK_K div 4) + 12], 2)
    when cpuEndian != littleEndian: dRaw = swapEndian(dRaw)
    let dAll = halfToFloat(dRaw)
    var scalesBytes: array[12, byte]
    copyMem(addr scalesBytes[0], addr src[offset + (QK_K div 8) + (QK_K div 4)], 12)
    var aux: array[4, uint32]
    for i in 0 ..< 3:
      let b = i * 4
      aux[i] = uint32(scalesBytes[b]) or (uint32(scalesBytes[b+1]) shl 8) or (uint32(scalesBytes[b+2]) shl 16) or (uint32(scalesBytes[b+3]) shl 24)
    let kmask1 = 0x03030303'u32
    let kmask2 = 0x0f0f0f0f'u32
    let tmp = aux[2]
    aux[2] = ((aux[0] shr 4) and kmask2) or (((tmp shr 4) and kmask1) shl 4)
    aux[3] = ((aux[1] shr 4) and kmask2) or (((tmp shr 6) and kmask1) shl 4)
    aux[0] = (aux[0] and kmask2) or (((tmp shr 0) and kmask1) shl 4)
    aux[1] = (aux[1] and kmask2) or (((tmp shr 2) and kmask1) shl 4)
    var scales: array[16, int8]
    for i in 0 ..< 4:
      let v = aux[i]
      scales[i*4+0] = int8(v and 0xFF)
      scales[i*4+1] = int8((v shr 8) and 0xFF)
      scales[i*4+2] = int8((v shr 16) and 0xFF)
      scales[i*4+3] = int8((v shr 24) and 0xFF)
    var scaleIdx = 0
    var m: uint8 = 1
    var qOffset = 0
    for _ in 0 ..< 2:
      var shift = 0
      for _ in 0 ..< 4:
        let dl = dAll * float32(scales[scaleIdx] - 32)
        inc scaleIdx
        for l in 0 ..< 16:
          let qv = int8((qs[qOffset + l] shr shift) and 3)
          let hm = (if (hmask[l] and m) != 0: 0 else: 4)
          dst[outIdx] = dl * float32(qv - hm)
          inc outIdx
        let dl2 = dAll * float32(scales[scaleIdx] - 32)
        inc scaleIdx
        for l in 0 ..< 16:
          let qv = int8((qs[qOffset + 16 + l] shr shift) and 3)
          let hm = (if (hmask[16 + l] and m) != 0: 0 else: 4)
          dst[outIdx] = dl2 * float32(qv - hm)
          inc outIdx
        shift += 2
        m = m shl 1
      qOffset += 32
    offset += blockQ3KSize

proc dequantRowQ4K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    var dRaw, dminRaw: uint16
    copyMem(addr dRaw, addr src[offset], 2)
    copyMem(addr dminRaw, addr src[offset + 2], 2)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    let scales = cast[ptr UncheckedArray[uint8]](addr src[offset + 4])
    let qs = cast[ptr UncheckedArray[uint8]](addr src[offset + 16])
    var scIdx = 0
    var qIdx = 0
    for j in 0 ..< 8:
      let sc = scales[scIdx]
      inc scIdx
      let dl = d * float32(sc and 0xF)
      let ml = dmin * float32(sc shr 4)
      for l in 0 ..< 32:
        let q = if (l < 16): (qs[qIdx + l] and 0xF) else: (qs[qIdx + l - 16] shr 4)
        dst[outIdx + l] = dl * float32(q) - ml
      outIdx += 32
      qIdx += (if j mod 2 == 1: 32 else: 0)
    offset += blockQ4KSize

proc dequantRowQ5K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    var dRaw, dminRaw: uint16
    copyMem(addr dRaw, addr src[offset], 2)
    copyMem(addr dminRaw, addr src[offset + 2], 2)
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    let scales = cast[ptr UncheckedArray[uint8]](addr src[offset + 4])
    let qs = cast[ptr UncheckedArray[uint8]](addr src[offset + 16])
    let qh = cast[ptr UncheckedArray[uint8]](addr src[offset + 16 + (QK_K div 2)])
    var scIdx = 0
    var qIdx = 0
    var hm: uint8 = 1
    for j in 0 ..< 8:
      let sc = scales[scIdx]
      inc scIdx
      let dl = d * float32(sc and 0xF)
      let ml = dmin * float32(sc shr 4)
      for l in 0 ..< 32:
        var q = if (l < 16): (qs[qIdx + l] and 0xF) else: (qs[qIdx + l - 16] shr 4)
        if (qh[l] and hm) != 0: q = q or 16
        dst[outIdx + l] = dl * float32(q) - ml
      outIdx += 32
      qIdx += (if j mod 2 == 1: 32 else: 0)
      hm = hm shl 1
    offset += blockQ5KSize

proc dequantRowQ6K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let ql = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qh = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 2)])
    let sc = cast[ptr UncheckedArray[int8]](addr src[offset + (QK_K div 2) + (QK_K div 4)])
    var dRaw: uint16
    copyMem(addr dRaw, addr src[offset + (QK_K div 2) + (QK_K div 4) + (QK_K div 16)], 2)
    when cpuEndian != littleEndian: dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)
    var qlOff = 0
    var qhOff = 0
    var scOff = 0
    for _ in 0 ..< 2:
      for l in 0 ..< 32:
        let scaleBlock = l div 16
        let q1 = int8((ql[qlOff + l + 0] and 0xF) or (((qh[qhOff + l] shr 0) and 3) shl 4)) - 32
        let q2 = int8((ql[qlOff + l + 32] and 0xF) or (((qh[qhOff + l] shr 2) and 3) shl 4)) - 32
        let q3 = int8((ql[qlOff + l + 0] shr 4) or (((qh[qhOff + l] shr 4) and 3) shl 4)) - 32
        let q4 = int8((ql[qlOff + l + 32] shr 4) or (((qh[qhOff + l] shr 6) and 3) shl 4)) - 32
        dst[outIdx + l + 0]   = d * float32(sc[scOff + scaleBlock + 0]) * float32(q1)
        dst[outIdx + l + 32]  = d * float32(sc[scOff + scaleBlock + 2]) * float32(q2)
        dst[outIdx + l + 64]  = d * float32(sc[scOff + scaleBlock + 4]) * float32(q3)
        dst[outIdx + l + 96]  = d * float32(sc[scOff + scaleBlock + 6]) * float32(q4)
      outIdx += 128
      qlOff += 64
      qhOff += 32
      scOff += 8
    offset += blockQ6KSize
