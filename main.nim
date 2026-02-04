## Minimal CLI example: load GGUF and tokenize a prompt.

import std/[os, strutils, sequtils]
import ./gguf_loader
import ./tokenizer
import ./model
import ./forward
import ./tensor

proc argmaxLast(logits: Tensor, nVocab: int): int32 =
  let seqLen = logits.shape[0]
  let base = (seqLen - 1) * nVocab
  var bestIdx = 0
  var bestVal = logits.data[base]
  for i in 1 ..< nVocab:
    let v = logits.data[base + i]
    if v > bestVal:
      bestVal = v
      bestIdx = i
  int32(bestIdx)

proc main() =
  if paramCount() < 2:
    echo "usage: tinylama <model.gguf> <prompt> [max_new] [--progress]"
    quit(1)

  let modelPath = paramStr(1)
  let prompt = paramStr(2)

  var gg = openGguf(modelPath)
  let vocab = loadVocab(gg)
  gg.close()

  let tokens = tokenize(vocab, prompt, addSpecial = true)
  echo "token count: ", tokens.len
  if tokens.len > 0:
    echo "tokens: ", tokens.mapIt($it).join(" ")

  var m = loadModel(modelPath)
  defer: m.close()
  let nVocab = m.hparams.nVocab
  var allTokens = tokens
  var maxNew = 4
  var showProgress = false
  if paramCount() >= 3:
    let arg3 = paramStr(3)
    if arg3 == "--progress":
      showProgress = true
    else:
      maxNew = parseInt(arg3)
  if paramCount() >= 4 and paramStr(4) == "--progress":
    showProgress = true
  var cache = initKvCache(m.hparams, tokens.len + maxNew)
  stdout.write(detokenize(vocab, tokens))
  if showProgress: stderr.write("prefill... ")
  let logits0 = forwardPrefill(m, allTokens, cache)
  if showProgress: stderr.writeLine("done")
  var next = argmaxLast(logits0, nVocab)
  allTokens.add(next)
  stdout.write(detokenize(vocab, @[next]))
  stdout.flushFile()
  for i in 1 ..< maxNew:
    if showProgress: stderr.write("decode ", $i, "/", $maxNew, "... ")
    let logits = forwardDecode(m, next, cache)
    next = argmaxLast(logits, nVocab)
    allTokens.add(next)
    stdout.write(detokenize(vocab, @[next]))
    stdout.flushFile()
    if showProgress: stderr.writeLine("tok=" & $next)
  stdout.write("\n")

when isMainModule:
  main()
