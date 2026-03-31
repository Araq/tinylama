## Minimal CLI example: load GGUF and tokenize a prompt.

import std/[os, strutils]
import ./gguf_loader
import ./tokenizer
import ./model
import ./forward
import ./infer_core
import ./tensor

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
    let tokens = encodePromptTokens(vocab, line)
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
