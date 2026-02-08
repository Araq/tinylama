import std/[strutils, times, random, algorithm, parseopt]
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
  var modelPath = ""
  var prompt = "Once upon a time"
  var maxNew = 128
  var temp = 0.8f
  var topK = 40
  var topP = 0.9f

  var p = initOptParser()
  var args: seq[string] = @[]
  while true:
    p.next()
    case p.kind
    of cmdEnd: break
    of cmdShortOption, cmdLongOption:
      let key = p.key
      var val = p.val
      if val == "":
        p.next()
        if p.kind == cmdArgument:
          val = p.key
        else:
          # If next isn't an argument, we might have a flag without value
          # but here all our options expect values.
          discard

      case key
      of "max-new", "n":
        if val != "": maxNew = parseInt(val)
      of "temp", "t":
        if val != "": temp = parseFloat(val)
      of "top-k", "k":
        if val != "": topK = parseInt(val)
      of "top-p", "p":
        if val != "": topP = parseFloat(val)
      else: discard
    of cmdArgument:
      args.add(p.key)

  if args.len < 1:
    echo "Usage: tinylama [options] <model.gguf> [prompt]"
    echo "Options:"
    echo "  --max-new, -n:<int>   Max new tokens (default: 128)"
    echo "  --temp, -t:<float>    Temperature (default: 0.8)"
    echo "  --top-k, -k:<int>     Top-K (default: 40)"
    echo "  --top-p, -p:<float>   Top-P (default: 0.9)"
    return

  modelPath = args[0]
  if args.len >= 2:
    prompt = args[1]

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
  let sampler = Sampler(temp: temp, topK: topK, topP: topP)

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
