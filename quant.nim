## Dequantization helpers for GGUF tensors.

import std/[math]
when cpuEndian != littleEndian:
  import std/endians

const
  QK_K* = 256
  blockQ2KSize* = 2 + 2 + (QK_K div 16) + (QK_K div 4) # d, dmin, scales, qs
  blockQ3KSize* = 2 + (QK_K div 4) + (QK_K div 8) + 12  # d, qs, hmask, scales
  blockQ6KSize* = 2 + (QK_K div 16) + 3 * (QK_K div 4) # d, scales, ql+qh

proc halfToFloat*(h: uint16): float32 =
  ## IEEE 754 half to float32.
  let s = (h shr 15) and 0x1
  let e = (h shr 10) and 0x1F
  let f = h and 0x3FF
  if e == 0:
    if f == 0:
      return (if s == 1: -0.0'f32 else: 0.0'f32)
    # subnormal
    return (if s == 1: -1.0'f32 else: 1.0'f32) *
      pow(2.0'f32, -14.0'f32) * (float32(f) / 1024.0'f32)
  elif e == 31:
    if f == 0:
      return (if s == 1: -Inf else: Inf)
    return NaN
  else:
    let exp = int(e) - 15
    let mant = 1.0'f32 + float32(f) / 1024.0'f32
    let sign = (if s == 1: -1.0'f32 else: 1.0'f32)
    return sign * mant * pow(2.0'f32, float32(exp))

proc dequantRowQ2K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  ## Dequantize a row of k elements (k must be multiple of QK_K).
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q2_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let scalesPtr = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qsPtr = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 16)])
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 16) + (QK_K div 4)])[0]
    var dminRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 16) + (QK_K div 4)])[1]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
      dminRaw = swapEndian(dminRaw)
    let d = halfToFloat(dRaw)
    let dmin = halfToFloat(dminRaw)
    var scaleIdx = 0
    var qOffset = 0
    for n in 0 ..< 2: # 2 chunks of 128
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
  ## Dequantize a row of k elements (k must be multiple of QK_K).
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q3_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let hmask = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qs = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 8)])
    var scalesBytes: array[12, byte]
    copyMem(addr scalesBytes[0], addr src[offset + (QK_K div 8) + (QK_K div 4)], 12)
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 8) + (QK_K div 4) + 12])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
    let dAll = halfToFloat(dRaw)

    var aux: array[4, uint32]
    for i in 0 ..< 3:
      let b = i * 4
      aux[i] = uint32(scalesBytes[b]) or (uint32(scalesBytes[b + 1]) shl 8) or
               (uint32(scalesBytes[b + 2]) shl 16) or (uint32(scalesBytes[b + 3]) shl 24)
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
      scales[i * 4 + 0] = int8((v shr 0) and 0xFF)
      scales[i * 4 + 1] = int8((v shr 8) and 0xFF)
      scales[i * 4 + 2] = int8((v shr 16) and 0xFF)
      scales[i * 4 + 3] = int8((v shr 24) and 0xFF)

    var scaleIdx = 0
    var m: uint8 = 1
    var qOffset = 0
    for _ in 0 ..< 2: # 2 chunks of 128
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

proc dequantRowQ6K*(src: ptr UncheckedArray[byte], dst: ptr UncheckedArray[float32], k: int) =
  ## Dequantize a row of k elements (k must be multiple of QK_K).
  if (k mod QK_K) != 0:
    raise newException(ValueError, "q6_K row size must be multiple of 256")
  let nBlocks = k div QK_K
  var outIdx = 0
  var offset = 0
  for _ in 0 ..< nBlocks:
    let ql = cast[ptr UncheckedArray[uint8]](addr src[offset])
    let qh = cast[ptr UncheckedArray[uint8]](addr src[offset + (QK_K div 2)])
    let sc = cast[ptr UncheckedArray[int8]](addr src[offset + (QK_K div 2) + (QK_K div 4)])
    var dRaw = cast[ptr UncheckedArray[uint16]](addr src[offset + (QK_K div 2) + (QK_K div 4) + (QK_K div 16)])[0]
    when cpuEndian != littleEndian:
      dRaw = swapEndian(dRaw)
    let d = halfToFloat(dRaw)

    var qlOff = 0
    var qhOff = 0
    var scOff = 0
    for _ in 0 ..< 2: # 2 chunks of 128
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

