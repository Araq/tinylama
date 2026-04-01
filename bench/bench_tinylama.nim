## Benchmarks for tinylama with Benchy.
##
## Usage:
##   nim c -r -d:release bench/bench_tinylama.nim models/model.gguf
##   nim c -r -d:release bench/bench_tinylama.nim models/model.gguf --decode-steps 32 --decode-warmup 2 --decode-runs 8

import
  std/[math, monotimes, os, strformat, strutils],
  benchy,
  ../src/[forward, gguf_loader, infer_core, model, tensor, tokenizer]

when defined(useHippo):
  import ../src/forward_hippo

const
  ShortPrompt = "Write one sentence about Nim."
  LongPrompt =
    "Explain transformers in simple terms, then give a compact example " &
    "of how token-by-token decoding works."
  DefaultDecodeSteps = 128
  DefaultDecodeWarmup = 1
  DefaultDecodeRuns = 5

  # Reference output for Q2_K model with ShortPrompt, 8 decode steps.
  # "\nNimi is a small town in the"
  ExpectedFirstToken: int32 = 13
  ExpectedTokens: array[8, int32] = [
    29940'i32, 10233, 338, 263, 2319, 4726, 297, 278
  ]

proc meanValue(values: seq[float64]): float64 =
  ## Return the arithmetic mean from a non-empty float sequence.
  var total = 0.0
  for value in values:
    total += value
  result = total / values.len.float64

proc stdevValue(values: seq[float64]): float64 =
  ## Sample standard deviation (Bessel-corrected), matching llama-bench.
  if values.len <= 1: return 0.0
  let mean = meanValue(values)
  var sqSum = 0.0
  for v in values: sqSum += v * v
  let n = values.len.float64
  result = sqrt(sqSum / (n - 1.0) - mean * mean * n / (n - 1.0))

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
  when defined(useHippo):
    result.gpuCache = initGpuKvCache(src.k.len, src.nHeadKv, src.headDim, src.maxLen)
    result.gpuCache.curLen = src.gpuCache.curLen
    let stream = gpuCtx.stream
    for i in 0 ..< src.k.len:
      let bytes = src.k[i].data.len * sizeof(float32)
      gpuMemcpyDevice(result.gpuCache.k[i].devicePtr, src.gpuCache.k[i].devicePtr, bytes, stream)
      gpuMemcpyDevice(result.gpuCache.v[i].devicePtr, src.gpuCache.v[i].devicePtr, bytes, stream)
    gpuStreamSync(stream)

proc runDecodeSteps(
  m: var Model,
  decodeBase: KvCache,
  firstGenerated: int32,
  nVocab, decodeSteps: int
): seq[int32] =
  ## Run decode for a fixed token count. Returns generated token IDs.
  var cache = cloneCache(decodeBase)
  var next = firstGenerated
  result = newSeq[int32](decodeSteps)
  for i in 0 ..< decodeSteps:
    let logits = forwardDecode(m, next, cache)
    next = argmaxLast(logits, nVocab)
    result[i] = next

proc measureDecodeSamples(
  m: var Model,
  decodeBase: KvCache,
  firstGenerated: int32,
  nVocab, decodeSteps, decodeRuns: int
): seq[float64] =
  ## Measure decode tok/s per sample (cache clone excluded from timing).
  result = newSeq[float64](decodeRuns)
  for i in 0 ..< decodeRuns:
    var cache = cloneCache(decodeBase)
    var next = firstGenerated
    let startNs = getMonoTime().ticks
    for j in 0 ..< decodeSteps:
      let logits = forwardDecode(m, next, cache)
      next = argmaxLast(logits, nVocab)
    let elapsedNs = (getMonoTime().ticks - startNs).float64
    result[i] = 1e9 * decodeSteps.float64 / elapsedNs

proc printDecodeSummary(tsSamples: seq[float64], decodeSteps: int) =
  ## Print decode throughput: avg ± stdev tok/s (matching llama-bench style).
  let
    avgTs = meanValue(tsSamples)
    stdTs = stdevValue(tsSamples)
  echo "decode summary:"
  echo fmt"  samples:   {tsSamples.len}"
  echo fmt"  n_tokens:  {decodeSteps}"
  echo fmt"  avg tok/s: {avgTs:.2f} ± {stdTs:.2f}"

proc main() =
  ## Run benchmark cases with steady-state decode throughput reporting.
  if paramCount() < 1 or paramStr(1).startsWith("--"):
    echo "usage: bench_tinylama <model.gguf> [--decode-steps N] [--decode-warmup N] [--decode-runs N]"
    quit(1)

  let modelPath = paramStr(1)
  var
    decodeSteps = DefaultDecodeSteps
    decodeWarmup = DefaultDecodeWarmup
    decodeRuns = DefaultDecodeRuns

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

  let shortTokens = encodePromptTokens(vocab, ShortPrompt)
  let longTokens = encodePromptTokens(vocab, LongPrompt)

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
    discard encodePromptTokens(vocab, ShortPrompt)

  timeIt "tokenize long prompt":
    discard encodePromptTokens(vocab, LongPrompt)

  timeIt "load model metadata":
    var lm = loadModel(modelPath)
    lm.close()

  timeIt "forward prefill short":
    var cache = initKvCache(m.hparams, max(shortTokens.len + decodeSteps + 8, 128))
    discard forwardPrefill(m, shortTokens, cache)

  timeIt "forward decode 1 token":
    var cache = cloneCache(decodeBase)
    discard forwardDecode(m, firstGenerated, cache)

  # Validate output correctness against known-good reference (8 decode steps)
  block:
    let validationTokens = runDecodeSteps(m, decodeBase, firstGenerated, nVocab, 8)
    let text = detokenize(vocab, @[firstGenerated] & validationTokens)
    echo "generated: ", text
    if firstGenerated != ExpectedFirstToken:
      echo "VALIDATION FAILED: firstGenerated = ", firstGenerated, ", expected ", ExpectedFirstToken
      quit(1)
    for i in 0 ..< 8:
      if validationTokens[i] != ExpectedTokens[i]:
        echo "VALIDATION FAILED at token ", i, ": got ", validationTokens[i], ", expected ", ExpectedTokens[i]
        echo "  full expected: ", ExpectedTokens
        echo "  full got:      ", validationTokens
        quit(1)
    echo "output validation passed"

  echo "warming decode benchmark..."
  for _ in 0 ..< decodeWarmup:
    discard runDecodeSteps(m, decodeBase, firstGenerated, nVocab, decodeSteps)

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
