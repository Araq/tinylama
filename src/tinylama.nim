import std/[os, strutils, times, random, algorithm]
import ./model
import ./tokenizer
import ./forward
import ./tensor
import arraymancer

type
  Sampler = object
    temp: float32
    topK: int
    topP: float32

proc sample(logits: GGTensor, s: Sampler): int =
  var tmpCpu = logits.at.toCpu()
  let n = tmpCpu.shape[0]

  if s.temp > 0:
    for i in 0 ..< n:
      tmpCpu[i, 0] /= s.temp

  # Softmax
  var maxL = tmpCpu[0, 0]
  for i in 1 ..< n:
    if tmpCpu[i, 0] > maxL: maxL = tmpCpu[i, 0]

  var sum = 0.0f
  for i in 0 ..< n:
    tmpCpu[i, 0] = exp(tmpCpu[i, 0] - maxL)
    sum += tmpCpu[i, 0]

  for i in 0 ..< n:
    tmpCpu[i, 0] /= sum

  # Top-K
  var pairs = newSeq[(float32, int)](n)
  for i in 0 ..< n:
    pairs[i] = (tmpCpu[i, 0], i)

  if s.topK > 0 and s.topK < n:
    pairs.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
    for i in s.topK ..< pairs.len:
      pairs[i][0] = 0.0f

  # Top-P
  if s.topP > 0 and s.topP < 1.0f:
    pairs.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
    var cumsum = 0.0f
    var last = pairs.len - 1
    for i in 0 ..< pairs.len:
      cumsum += pairs[i][0]
      if cumsum > s.topP:
        last = i
        break
    for i in last + 1 ..< pairs.len:
      pairs[i][0] = 0.0f

  # Rescale
  sum = 0.0f
  for i in 0 ..< pairs.len: sum += pairs[i][0]
  if sum == 0: return pairs[0][1]
  for i in 0 ..< pairs.len: pairs[i][0] /= sum

  # Random sample
  let r = rand(1.0f)
  var acc = 0.0f
  for i in 0 ..< pairs.len:
    acc += pairs[i][0]
    if r <= acc: return pairs[i][1]
  return pairs[0][1]

proc main() =
  if paramCount() < 1:
    echo "Usage: tinylama <model.gguf> [prompt] [max-new-tokens]"
    return

  let modelPath = paramStr(1)
  let prompt = if paramCount() >= 2: paramStr(2) else: "Once upon a time"
  let maxNew = if paramCount() >= 3: parseInt(paramStr(3)) else: 128

  randomize()

  echo "Loading model: ", modelPath
  var m = loadModel(modelPath)
  echo "Architecture: ", m.hparams.arch
  echo "Params: ", m.hparams

  echo "Loading tokenizer..."
  let vocab = loadVocab(m.gguf)

  echo "Tokenizing..."
  let tokens = tokenize(vocab, prompt)
  echo "Prompt tokens: ", tokens

  var cache = initKvCache(m, 2048)
  let sampler = Sampler(temp: 0.8f, topK: 40, topP: 0.9f)

  echo "Prefill..."
  let startPrefill = cpuTime()
  var logits = forwardPrefill(m, cache, tokens)
  echo "Prefill done in ", (cpuTime() - startPrefill).formatFloat(ffDecimal, 2), "s"

  var next = int32(sample(logits, sampler))
  stdout.write(detokenize(vocab, @[next]))
  stdout.flushFile()

  for i in 1 ..< maxNew:
    if next == vocab.eosId: break
    logits = forwardNext(m, cache, next)
    next = int32(sample(logits, sampler))
    stdout.write(detokenize(vocab, @[next]))
    stdout.flushFile()
  echo ""

main()
