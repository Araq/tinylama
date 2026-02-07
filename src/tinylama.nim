## Minimal CLI example: load GGUF and tokenize a prompt.

import std/[os, strutils, math, algorithm, random]
import arraymancer
import ./gguf_loader
import ./tokenizer
import ./model
import ./forward
import ./tensor

proc sample(logits: GGTensor, nVocab: int, temperature: float32 = 0.0, topP: float32 = 1.0, topK: int = 50): int32 =
  let lshape = logits.shape
  let seqLen = if lshape[0] == nVocab: lshape[1] else: lshape[0]

  var vals = newSeq[float32](nVocab)
  if lshape[0] == nVocab:
    for i in 0 ..< nVocab: vals[i] = logits.at[i, seqLen - 1]
  else:
    for i in 0 ..< nVocab: vals[i] = logits.at[seqLen - 1, i]

  if temperature <= 0.0:
    var bestIdx = 0
    var bestVal = vals[0]
    for i in 1 ..< nVocab:
      if vals[i] > bestVal:
        bestVal = vals[i]
        bestIdx = i
    return int32(bestIdx)

  var maxLogit = vals[0]
  for i in 1 ..< nVocab: maxLogit = max(maxLogit, vals[i])
  var sumExp = 0.0'f32
  for i in 0 ..< nVocab:
    vals[i] = exp((vals[i] - maxLogit) / temperature)
    sumExp += vals[i]
  for i in 0 ..< nVocab: vals[i] /= sumExp

  type TokenProb = object
    id: int32
    p: float32
  var probs = newSeq[TokenProb](nVocab)
  for i in 0 ..< nVocab: probs[i] = TokenProb(id: int32(i), p: vals[i])

  # If we don't need topK or topP, just sample multinomial directly to save time
  if (topK <= 0 or topK >= nVocab) and topP >= 1.0:
    let r = rand(1.0'f32)
    var cur = 0.0'f32
    for i in 0 ..< nVocab:
      cur += vals[i]
      if r <= cur: return int32(i)
    return int32(nVocab - 1)

  # Apply Top-K
  if topK > 0 and topK < nVocab:
    probs.sort(proc (a, b: TokenProb): int = cmp(b.p, a.p))
    for i in topK ..< nVocab: probs[i].p = 0.0
    var newSum = 0.0'f32
    for i in 0 ..< topK: newSum += probs[i].p
    for i in 0 ..< topK: probs[i].p /= newSum

  # Apply Top-P
  if topP < 1.0:
    if topK <= 0 or topK >= nVocab:
      probs.sort(proc (a, b: TokenProb): int = cmp(b.p, a.p))
    var cumulativeP = 0.0'f32
    var lastIdx = nVocab - 1
    for i in 0 ..< nVocab:
      cumulativeP += probs[i].p
      if cumulativeP >= topP:
        lastIdx = i
        break
    var newSum = 0.0'f32
    for i in 0 .. lastIdx: newSum += probs[i].p
    let r = rand(newSum)
    var cur = 0.0'f32
    for i in 0 .. lastIdx:
      cur += probs[i].p
      if r <= cur: return probs[i].id
    return probs[0].id

  # Final fallback multinomial
  let r = rand(1.0'f32)
  var cur = 0.0'f32
  for i in 0 ..< probs.len:
    cur += probs[i].p
    if r <= cur: return probs[i].id
  return probs[^1].id

proc main() =
  randomize()
  if paramCount() < 1:
    echo "usage: tinylama <model.gguf> [prompt] [options]"
    echo "options:"
    echo "  --max-new N      Max new tokens (default: 16)"
    echo "  --temp T         Temperature (default: 0.0, greedy)"
    echo "  --top-p P        Top-P sampling (default: 1.0)"
    echo "  --top-k K        Top-K sampling (default: 50)"
    echo "  --progress       Show generation progress"
    quit(1)

  let modelPath = paramStr(1)
  var prompt = ""
  var startIdx = 2
  if paramCount() >= 2 and not paramStr(2).startsWith("--"):
    prompt = paramStr(2)
    startIdx = 3

  var maxNew = 16
  var temp: float32 = 0.0
  var topP: float32 = 1.0
  var topK = 50
  var showProgress = false
  var i = startIdx
  while i <= paramCount():
    let arg = paramStr(i)
    if arg == "--progress":
      showProgress = true
    elif arg == "--max-new" and i + 1 <= paramCount():
      maxNew = parseInt(paramStr(i + 1))
      inc i
    elif arg == "--temp" and i + 1 <= paramCount():
      temp = parseFloat(paramStr(i + 1)).float32
      inc i
    elif arg == "--top-p" and i + 1 <= paramCount():
      topP = parseFloat(paramStr(i + 1)).float32
      inc i
    elif arg == "--top-k" and i + 1 <= paramCount():
      topK = parseInt(paramStr(i + 1))
      inc i
    else:
      echo "unknown arg: ", arg
      quit(1)
    inc i

  var gg = openGguf(modelPath)
  let vocab = loadVocab(gg)
  gg.close()

  var m = loadModel(modelPath)
  defer: m.close()
  let nVocab = m.hparams.nVocab

  var allTokens: seq[int32] = @[]
  var cache = initKvCache(m.hparams, max(32, maxNew + 32))

  proc runPrompt(line: string) =
    let fullPrompt = formatChatPrompt(vocab, line)
    let useSpecial = fullPrompt != line
    let tokens = tokenizeWithSpecial(vocab, fullPrompt, addSpecial = not useSpecial)
    allTokens.add(tokens)

    if cache.curLen == 0:
      if showProgress: stderr.write("prefill... ")
      let logits0 = forwardPrefill(m, allTokens, cache)
      if showProgress: stderr.writeLine("done")
      var next = sample(logits0, nVocab, temp, topP, topK)
      allTokens.add(next)
      stdout.write(detokenize(vocab, @[next]))
      stdout.flushFile()
      for j in 1 ..< maxNew:
        if showProgress: stderr.write("decode ", $j, "/", $maxNew, "... ")
        let logits = forwardDecode(m, next, cache)
        next = sample(logits, nVocab, temp, topP, topK)
        allTokens.add(next)
        stdout.write(detokenize(vocab, @[next]))
        stdout.flushFile()
        if showProgress: stderr.writeLine("tok=" & $next)
      stdout.write("\n")
    else:
      var lastLogits: GGTensor
      for t in tokens:
        lastLogits = forwardDecode(m, t, cache)
      var next = sample(lastLogits, nVocab, temp, topP, topK)
      allTokens.add(next)
      stdout.write(detokenize(vocab, @[next]))
      stdout.flushFile()
      for j in 1 ..< maxNew:
        if showProgress: stderr.write("decode ", $j, "/", $maxNew, "... ")
        let logits = forwardDecode(m, next, cache)
        next = sample(logits, nVocab, temp, topP, topK)
        allTokens.add(next)
        stdout.write(detokenize(vocab, @[next]))
        stdout.flushFile()
        if showProgress: stderr.writeLine("tok=" & $next)
      stdout.write("\n")

  if prompt.len > 0:
    runPrompt(prompt)
    return

  echo "REPL ready. Enter text, empty line or :quit to exit."
  while true:
    stdout.write("> ")
    stdout.flushFile()
    var line = ""
    if not stdin.readLine(line):
      break
    line = line.strip()
    if line.len == 0 or line == ":quit" or line == ":q":
      break
    runPrompt(line)

when isMainModule:
  main()
