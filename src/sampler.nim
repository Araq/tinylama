## Temperature + top-p (nucleus) sampling with repetition penalty
## over the last logits position.

import std/[random, math, algorithm, sets]
import ./tensor

type
  Sampler* = object
    temperature*: float32   ## 0 means greedy decoding
    topP*: float32          ## 1.0 disables nucleus filtering
    repeatPenalty*: float32 ## 1.0 disables; typical 1.1
    repeatWindow*: int      ## how many recent tokens are penalized
    rng: Rand

proc initSampler*(temperature = 0.7'f32, topP = 0.9'f32, seed = 0'i64,
                  repeatPenalty = 1.1'f32, repeatWindow = 256): Sampler =
  result.temperature = temperature
  result.topP = topP
  result.repeatPenalty = repeatPenalty
  result.repeatWindow = repeatWindow
  result.rng = initRand(if seed == 0: 0x1337'i64 else: seed)

proc applyRepeatPenalty(s: Sampler, row: var seq[float32],
                        recent: openArray[int32]) =
  ## llama.cpp-style: each unique recent token's logit is divided by the
  ## penalty when positive, multiplied when negative.
  if s.repeatPenalty == 1.0'f32 or recent.len == 0:
    return
  var seen = initHashSet[int32]()
  let start = max(0, recent.len - s.repeatWindow)
  for i in start ..< recent.len:
    let t = recent[i]
    if int(t) < row.len and not seen.containsOrIncl(t):
      if row[t] > 0:
        row[t] = row[t] / s.repeatPenalty
      else:
        row[t] = row[t] * s.repeatPenalty

proc lastLogits*(logits: Tensor, nVocab: int): seq[float32] =
  ## Extract the logits of the final sequence position, for either
  ## [nVocab, seqLen] or [seqLen, nVocab] layout.
  if logits.shape.len != 2:
    raise newException(ValueError, "logits must be 2D")
  result = newSeq[float32](nVocab)
  if logits.shape[0] == nVocab:
    let seqLen = logits.shape[1]
    let col = seqLen - 1
    for i in 0 ..< nVocab:
      result[i] = logits.data[i * seqLen + col]
  elif logits.shape[1] == nVocab:
    let base = (logits.shape[0] - 1) * nVocab
    for i in 0 ..< nVocab:
      result[i] = logits.data[base + i]
  else:
    raise newException(ValueError, "logits shape mismatch for vocab")

proc sample*(s: var Sampler, logits: Tensor, nVocab: int,
             recent: openArray[int32] = []): int32 =
  var row = lastLogits(logits, nVocab)
  applyRepeatPenalty(s, row, recent)
  if s.temperature <= 0.0'f32:
    var best = 0
    for i in 1 ..< nVocab:
      if row[i] > row[best]: best = i
    return int32(best)

  var maxVal = row[0]
  for i in 1 ..< nVocab:
    if row[i] > maxVal: maxVal = row[i]
  var probs = newSeq[(float32, int32)](nVocab)
  var total = 0.0'f32
  for i in 0 ..< nVocab:
    let p = exp((row[i] - maxVal) / s.temperature)
    probs[i] = (p, int32(i))
    total += p

  probs.sort(proc (a, b: (float32, int32)): int =
    if a[0] > b[0]: -1 elif a[0] < b[0]: 1 else: 0)

  var cutoff = probs.len
  if s.topP < 1.0'f32:
    var acc = 0.0'f32
    let target = s.topP * total
    for i in 0 ..< probs.len:
      acc += probs[i][0]
      if acc >= target:
        cutoff = i + 1
        break

  var mass = 0.0'f32
  for i in 0 ..< cutoff:
    mass += probs[i][0]
  var r = s.rng.rand(1.0).float32 * mass
  for i in 0 ..< cutoff:
    r -= probs[i][0]
    if r <= 0.0'f32:
      return probs[i][1]
  probs[cutoff - 1][1]
