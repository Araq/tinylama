## Minimal CLI example: load GGUF and tokenize a prompt.

import std/[os, strutils]
import ./gguf_loader
import ./tokenizer
import ./model
import ./forward
import ./tensor

proc argmaxLast(logits: Tensor, nVocab: int): int32 =
  if logits.shape.len != 2:
    raise newException(ValueError, "logits must be 2D")
  if logits.shape[0] == nVocab:
    let seqLen = logits.shape[1]
    let col = seqLen - 1
    var bestIdx = 0
    var bestVal = logits.data[col]
    for i in 1 ..< nVocab:
      let v = logits.data[i * seqLen + col]
      if v > bestVal:
        bestVal = v
        bestIdx = i
    return int32(bestIdx)
  if logits.shape[1] == nVocab:
    let seqLen = logits.shape[0]
    let base = (seqLen - 1) * nVocab
    var bestIdx = 0
    var bestVal = logits.data[base]
    for i in 1 ..< nVocab:
      let v = logits.data[base + i]
      if v > bestVal:
        bestVal = v
        bestIdx = i
    return int32(bestIdx)
  raise newException(ValueError, "logits shape mismatch for vocab")

proc main() =
  if paramCount() < 1:
    echo "usage: tinylama <model.gguf> [prompt] [--max-new N] [--progress]"
    quit(1)

  let modelPath = paramStr(1)
  var prompt = ""
  var startIdx = 2
  if paramCount() >= 2 and not paramStr(2).startsWith("--"):
    prompt = paramStr(2)
    startIdx = 3

  var maxNew = 4
  var showProgress = false
  var i = startIdx
  while i <= paramCount():
    let arg = paramStr(i)
    if arg == "--progress":
      showProgress = true
    elif arg == "--max-new" and i + 1 <= paramCount():
      maxNew = parseInt(paramStr(i + 1))
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
      var next = argmaxLast(logits0, nVocab)
      allTokens.add(next)
      stdout.write(detokenize(vocab, @[next]))
      stdout.flushFile()
      for j in 1 ..< maxNew:
        if showProgress: stderr.write("decode ", $j, "/", $maxNew, "... ")
        let logits = forwardDecode(m, next, cache)
        next = argmaxLast(logits, nVocab)
        allTokens.add(next)
        stdout.write(detokenize(vocab, @[next]))
        stdout.flushFile()
        if showProgress: stderr.writeLine("tok=" & $next)
      stdout.write("\n")
    else:
      var lastLogits: Tensor
      for t in tokens:
        lastLogits = forwardDecode(m, t, cache)
      var next = argmaxLast(lastLogits, nVocab)
      allTokens.add(next)
      stdout.write(detokenize(vocab, @[next]))
      stdout.flushFile()
      for j in 1 ..< maxNew:
        if showProgress: stderr.write("decode ", $j, "/", $maxNew, "... ")
        let logits = forwardDecode(m, next, cache)
        next = argmaxLast(logits, nVocab)
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
