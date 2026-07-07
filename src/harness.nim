## Agentic harness dedicated to Nim: drives a local chat model (Qwen-style
## ChatML) and lets it navigate Nim code via nimony's one-shot IDE tools.
##
##   nim c -r src/harness.nim model.gguf "where is proc foo defined?" \
##     [--max-new N] [--temp T] [--top-p P] [--seed N] [--ctx N] \
##     [--max-turns N] [--nimony PATH] [--verbose]

import std/[os, strutils, json]
import ./gguf_loader
import ./tokenizer
import ./model
import ./forward
import ./tensor
import ./sampler
import ./chat
import ./nimtools
import ./toolconfig

type
  Harness = object
    m: Model
    vocab: Vocab
    cache: KvCache
    smp: Sampler
    stopIds: seq[int32]
    maxNew: int
    maxContinue: int
    verbose: bool
    history: seq[int32]  ## all tokens in context, for the repetition window
    pendingTok: int32    ## sampled but not yet emitted when a round ends
    hasPending: bool

proc findNimony(): string =
  result = getEnv("NIMONY")
  if result.len > 0: return
  result = findExe("nimony")
  if result.len > 0: return
  let guess = getHomeDir() / "projects" / "nimony" / "bin" / "nimony"
  if fileExists(guess): return guess
  result = ""

proc resetContext(h: var Harness) =
  ## Drop the KV cache so a compacted conversation can be re-prefilled;
  ## prefill rewrites the cache from position 0 on both backends.
  h.cache.curLen = 0
  h.history.setLen 0
  h.hasPending = false

proc feedTokens(h: var Harness, tokens: seq[int32]): Tensor =
  ## Append tokens to the context; returns the logits of the last position.
  if h.cache.curLen + tokens.len + h.maxNew > h.cache.maxLen:
    quit "context window exhausted (" & $h.cache.maxLen &
         " tokens); retry with a larger --ctx"
  h.history.add tokens
  if h.cache.curLen == 0:
    result = forwardPrefill(h.m, tokens, h.cache)
  else:
    for t in tokens:
      result = forwardDecode(h.m, t, h.cache)

proc recentWindow(h: Harness): seq[int32] =
  let start = max(0, h.history.len - h.smp.repeatWindow)
  h.history[start .. ^1]

proc generateTurn(h: var Harness, logits: Tensor): tuple[text: string, stopped: bool] =
  ## Sample until a stop token or the per-round token budget. When the
  ## budget runs out mid-turn the sampled-but-unprocessed token is kept, so
  ## a subsequent call resumes the same assistant turn seamlessly.
  result = (text: "", stopped: true)
  let nVocab = h.m.hparams.nVocab
  var next: int32
  if h.hasPending:
    next = h.pendingTok
    h.hasPending = false
  else:
    next = sample(h.smp, logits, nVocab, recentWindow(h))
  var produced = 0
  while next notin h.stopIds:
    if produced >= h.maxNew:
      h.pendingTok = next
      h.hasPending = true
      result.stopped = false
      break
    if h.cache.curLen >= h.cache.maxLen - 1:
      stderr.writeLine "\nwarning: context window exhausted mid-generation"
      break
    let piece = detokenize(h.vocab, @[next])
    stdout.write(piece)
    stdout.flushFile()
    result.text.add piece
    h.history.add next
    let l = forwardDecode(h.m, next, h.cache)
    next = sample(h.smp, l, nVocab, recentWindow(h))
    inc produced
  if result.stopped:
    stdout.write("\n")

proc runTask(h: var Harness, task: string, maxTurns: int,
             tools: seq[ToolDef], nimonyBin: string, toolTimeout: int) =
  var messages = @[
    msg(roleSystem, systemPrompt(tools)),
    msg(roleUser, task)
  ]
  var chunk = renderChatML(messages)
  var firstChunk = true

  for turn in 1 .. maxTurns:
    if h.verbose:
      stderr.writeLine "--- feeding chunk ---\n" & chunk & "\n---"
    var tokens = tokenizeWithSpecial(h.vocab, chunk, addSpecial = firstChunk)
    if h.cache.curLen + tokens.len + h.maxNew > h.cache.maxLen:
      # The next chunk would leave no room to generate: elide old tool
      # results and re-prefill the rebuilt conversation from scratch.
      for keepRecent in [2, 1, 0]:
        if elideOldToolResults(messages, keepRecent):
          h.resetContext()
          chunk = renderChatML(messages)
          tokens = tokenizeWithSpecial(h.vocab, chunk, addSpecial = true)
          if h.cache.curLen + tokens.len + h.maxNew <= h.cache.maxLen: break
      if h.cache.curLen + tokens.len + h.maxNew > h.cache.maxLen:
        quit "context window exhausted (" & $h.cache.maxLen &
             " tokens) even after eliding old tool results; retry with a larger --ctx"
      stderr.writeLine "[ctx] compacted conversation; re-prefilling " &
        $tokens.len & " tokens"
    firstChunk = false
    let logits = feedTokens(h, tokens)

    var answer = ""
    var continues = 0
    while true:
      let (text, stopped) = generateTurn(h, logits)
      answer.add text
      if stopped:
        break
      if continues >= h.maxContinue:
        stderr.writeLine "\nwarning: --max-new budget exceeded " &
          $h.maxContinue & " times; closing the turn"
        h.hasPending = false
        break
      inc continues

    var parseErrors: seq[string] = @[]
    let calls = parseToolCalls(answer, parseErrors)
    if calls.len == 0 and parseErrors.len == 0:
      return
    messages.add msg(roleAssistant, answer)
    var toolMsgs: seq[ChatMessage] = @[]
    for err in parseErrors:
      stderr.writeLine "[tool] parse error: " & err
      toolMsgs.add msg(roleTool, "error: " & err &
        ". Re-emit the call as <tool_call>{\"name\": \"<tool-name>\", " &
        "\"arguments\": { ... }}</tool_call> with strictly valid JSON.")
    for tc in calls:
      stderr.writeLine "[tool] " & tc.name & " " & $tc.arguments
      let response = execTool(tc, tools, nimonyBin, toolTimeout)
      stderr.writeLine "[tool] -> " & response.replace("\n", "\n[tool]    ")
      toolMsgs.add msg(roleTool, response)
    messages.add toolMsgs
    # Close the assistant turn (the sampled stop token was never fed to the
    # model) and append the tool results plus a fresh generation prompt.
    chunk = "<|im_end|>\n" & renderChatML(toolMsgs)
  stderr.writeLine "warning: gave up after " & $maxTurns & " turns"

proc main() =
  if paramCount() < 2:
    echo "usage: harness <model.gguf> <task> [--max-new N] [--max-continue N] " &
         "[--temp T] [--top-p P] [--repeat-penalty P] [--repeat-window N] " &
         "[--seed N] [--ctx N] [--max-turns N] [--tool-timeout SECS] " &
         "[--nimony PATH] [--tools FILE.nif] [--verbose]"
    quit(1)

  let modelPath = paramStr(1)
  let task = paramStr(2)
  var maxNew = 512
  var maxContinue = 4
  var temp = 0.7'f32
  var topP = 0.9'f32
  var repeatPenalty = 1.1'f32
  var repeatWindow = 256
  var seed = 0'i64
  var ctxLen = 4096
  var maxTurns = 8
  var toolTimeout = 60
  var nimonyBin = ""
  var toolsFile = ""
  var verbose = false

  var i = 3
  while i <= paramCount():
    let arg = paramStr(i)
    template nextVal(): string =
      inc i
      if i > paramCount(): quit "missing value for " & arg
      paramStr(i)
    case arg
    of "--max-new": maxNew = parseInt(nextVal())
    of "--max-continue": maxContinue = parseInt(nextVal())
    of "--temp": temp = parseFloat(nextVal()).float32
    of "--top-p": topP = parseFloat(nextVal()).float32
    of "--repeat-penalty": repeatPenalty = parseFloat(nextVal()).float32
    of "--repeat-window": repeatWindow = parseInt(nextVal())
    of "--seed": seed = parseBiggestInt(nextVal())
    of "--ctx": ctxLen = parseInt(nextVal())
    of "--max-turns": maxTurns = parseInt(nextVal())
    of "--tool-timeout": toolTimeout = parseInt(nextVal())
    of "--nimony": nimonyBin = nextVal()
    of "--tools": toolsFile = nextVal()
    of "--verbose": verbose = true
    else: quit "unknown arg: " & arg
    inc i

  if nimonyBin.len == 0:
    nimonyBin = findNimony()
  if nimonyBin.len == 0:
    quit "nimony binary not found; pass --nimony PATH or set NIMONY"

  var tools = builtinTools()
  if toolsFile.len == 0 and fileExists("harness.tools.nif"):
    toolsFile = "harness.tools.nif"
  if toolsFile.len > 0:
    for t in loadToolConfig(toolsFile):
      tools.addOrReplace t
    stderr.writeLine "[tools] " & $tools.len & " tools loaded"

  var gg = openGguf(modelPath)
  let vocab = loadVocab(gg)
  gg.close()

  var h = Harness(
    vocab: vocab,
    smp: initSampler(temp, topP, seed, repeatPenalty, repeatWindow),
    stopIds: stopTokenIds(vocab),
    maxNew: maxNew,
    maxContinue: maxContinue,
    verbose: verbose)
  h.m = loadModel(modelPath)
  defer: h.m.close()
  if h.m.hparams.nCtx > 0:
    ctxLen = min(ctxLen, h.m.hparams.nCtx)
  h.cache = initKvCache(h.m.hparams, ctxLen)

  runTask(h, task, maxTurns, tools, nimonyBin, toolTimeout)

when isMainModule:
  main()
