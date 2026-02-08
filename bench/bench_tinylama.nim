## Benchmarks for tinylama with Benchy.
##
## Usage:
##   nim c -r -d:release bench/bench_tinylama.nim models/model.gguf

import
  std/[os],
  benchy,
  ../src/[forward, gguf_loader, infer_core, model, tensor, tokenizer]

const
  shortPrompt = "Write one sentence about Nim."
  longPrompt =
    "Explain transformers in simple terms, then give a compact example " &
    "of how token-by-token decoding works."
  decodeSteps = 32

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

proc main() =
  ## Run benchmark cases for tokenization, prefill, and decode throughput.
  if paramCount() < 1:
    echo "usage: bench_tinylama <model.gguf>"
    quit(1)

  let modelPath = paramStr(1)

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

  timeIt "forward decode 32 tokens":
    var cache = cloneCache(decodeBase)
    var next = firstGenerated
    for _ in 0 ..< decodeSteps:
      let logits = forwardDecode(m, next, cache)
      next = argmaxLast(logits, nVocab)

when isMainModule:
  main()
