## Benchmarks for tinylama with Benchy.
##
## Usage:
##   nim c -r -d:release bench/bench_tinylama.nim models/model.gguf
##   nim c -r -d:release bench/bench_tinylama.nim models/model.gguf --decode-steps 32 --decode-warmup 2 --decode-runs 8

import
  std/[monotimes, os, strformat, strutils],
  benchy,
  ../src/[forward, gguf_loader, infer_core, model, tensor, tokenizer]

const
  shortPrompt = "Write one sentence about Nim."
  longPrompt =
    "Explain transformers in simple terms, then give a compact example " &
    "of how token-by-token decoding works."
  defaultDecodeSteps = 32
  defaultDecodeWarmup = 1
  defaultDecodeRuns = 4

proc nowMs(): float64 =
  ## Return the current monotonic time in milliseconds.
  getMonoTime().ticks.float64 / 1_000_000.0

proc minValue(values: seq[float64]): float64 =
  ## Return the minimum value from a non-empty float sequence.
  result = values[0]
  for i in 1 ..< values.len:
    if values[i] < result:
      result = values[i]

proc meanValue(values: seq[float64]): float64 =
  ## Return the arithmetic mean from a non-empty float sequence.
  var total = 0.0
  for value in values:
    total += value
  result = total / values.len.float64

proc parsePositiveInt(value, flagName: string): int =
  ## Parse an integer CLI value and ensure it is at least 1.
  result = parseInt(value)
  if result < 1:
    raise newException(ValueError, flagName & " must be >= 1")

proc cloneCache(src: KvCache): KvCache =
  ## Deep-copy the KV cache so each benchmark starts from the same state.
  result.curLen = src.curLen
  result.maxLen = src.maxLen
  result.nHeadKv = src.nHeadKv
  result.headDim = src.headDim
  result.k = newSeq[Tensor](src.k.len)
  result.v = newSeq[Tensor](src.v.len)
  for i in 0 ..< src.k.len:
    result.k[i] = newTensor(src.k[i].shape)
    result.v[i] = newTensor(src.v[i].shape)
    for j in 0 ..< src.k[i].data.len:
      result.k[i].data[j] = src.k[i].data[j]
    for j in 0 ..< src.v[i].data.len:
      result.v[i].data[j] = src.v[i].data[j]

proc runDecodeSteps(
  m: var Model,
  decodeBase: KvCache,
  firstGenerated: int32,
  nVocab, decodeSteps: int
) =
  ## Run decode for a fixed token count from the same seeded cache state.
  var cache = cloneCache(decodeBase)
  var next = firstGenerated
  for _ in 0 ..< decodeSteps:
    let logits = forwardDecode(m, next, cache)
    next = argmaxLast(logits, nVocab)

proc measureDecodeSamples(
  m: var Model,
  decodeBase: KvCache,
  firstGenerated: int32,
  nVocab, decodeSteps, decodeRuns: int
): seq[float64] =
  ## Measure steady-state decode timings with one full decode run per sample.
  result = newSeq[float64](decodeRuns)
  for i in 0 ..< decodeRuns:
    let startMs = nowMs()
    runDecodeSteps(m, decodeBase, firstGenerated, nVocab, decodeSteps)
    result[i] = nowMs() - startMs

proc printDecodeSummary(samples: seq[float64], decodeSteps: int) =
  ## Print compact decode throughput metrics from collected timing samples.
  let
    minMs = minValue(samples)
    avgMs = meanValue(samples)
    minMsPerToken = minMs / decodeSteps.float64
    avgMsPerToken = avgMs / decodeSteps.float64
    minTokPerSec = 1000.0 / minMsPerToken
    avgTokPerSec = 1000.0 / avgMsPerToken
  echo "decode summary:"
  echo fmt"  best total: {minMs:.3f} ms"
  echo fmt"  mean total: {avgMs:.3f} ms"
  echo fmt"  best ms/token: {minMsPerToken:.3f}"
  echo fmt"  mean ms/token: {avgMsPerToken:.3f}"
  echo fmt"  best tok/s: {minTokPerSec:.3f}"
  echo fmt"  mean tok/s: {avgTokPerSec:.3f}"

proc main() =
  ## Run benchmark cases with steady-state decode throughput reporting.
  if paramCount() < 1 or paramStr(1).startsWith("--"):
    echo "usage: bench_tinylama <model.gguf> [--decode-steps N] [--decode-warmup N] [--decode-runs N]"
    quit(1)

  let modelPath = paramStr(1)
  var
    decodeSteps = defaultDecodeSteps
    decodeWarmup = defaultDecodeWarmup
    decodeRuns = defaultDecodeRuns

  var i = 2
  while i <= paramCount():
    let arg = paramStr(i)
    if arg == "--decode-steps" and i + 1 <= paramCount():
      decodeSteps = parsePositiveInt(paramStr(i + 1), "--decode-steps")
      inc i
    elif arg == "--decode-warmup" and i + 1 <= paramCount():
      decodeWarmup = parsePositiveInt(paramStr(i + 1), "--decode-warmup")
      inc i
    elif arg == "--decode-runs" and i + 1 <= paramCount():
      decodeRuns = parsePositiveInt(paramStr(i + 1), "--decode-runs")
      inc i
    else:
      echo "unknown arg: ", arg
      quit(1)
    inc i

  var gg = openGguf(modelPath)
  let vocab = loadVocab(gg)
  gg.close()

  let shortTokens = encodePromptTokens(vocab, shortPrompt)
  let longTokens = encodePromptTokens(vocab, longPrompt)

  var m = loadModel(modelPath)
  defer: m.close()
  let nVocab = m.hparams.nVocab

  var seedCache = initKvCache(m.hparams, max(shortTokens.len + decodeSteps + 8, 128))
  let prefillSeedLogits = forwardPrefill(m, shortTokens, seedCache)
  let firstGenerated = argmaxLast(prefillSeedLogits, nVocab)
  let decodeBase = cloneCache(seedCache)

  echo "model: ", modelPath
  echo "short prompt tokens: ", shortTokens.len
  echo "long prompt tokens:  ", longTokens.len
  echo "decode steps:        ", decodeSteps
  echo "decode warmup runs:  ", decodeWarmup
  echo "decode sample runs:  ", decodeRuns

  timeIt "tokenize short prompt":
    discard encodePromptTokens(vocab, shortPrompt)

  timeIt "tokenize long prompt":
    discard encodePromptTokens(vocab, longPrompt)

  timeIt "load model metadata":
    var lm = loadModel(modelPath)
    lm.close()

  timeIt "forward prefill short":
    var cache = initKvCache(m.hparams, max(shortTokens.len + decodeSteps + 8, 128))
    discard forwardPrefill(m, shortTokens, cache)

  timeIt "forward decode 1 token":
    var cache = cloneCache(decodeBase)
    discard forwardDecode(m, firstGenerated, cache)

  echo "warming decode benchmark..."
  for _ in 0 ..< decodeWarmup:
    runDecodeSteps(m, decodeBase, firstGenerated, nVocab, decodeSteps)

  echo "running decode samples..."
  let decodeSamples = measureDecodeSamples(
    m,
    decodeBase,
    firstGenerated,
    nVocab,
    decodeSteps,
    decodeRuns
  )
  printDecodeSummary(decodeSamples, decodeSteps)

when isMainModule:
  main()
