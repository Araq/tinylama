## Minimal tokenizer for GGUF models (SPM/LLaMA + basic GPT2 byte-BPE IO).

import std/[tables, heapqueue, strutils, sequtils, unicode]
import ./gguf_loader

const
  tokenModelKey = "tokenizer.ggml.model"
  tokenListKey = "tokenizer.ggml.tokens"
  tokenScoresKey = "tokenizer.ggml.scores"
  tokenTypesKey = "tokenizer.ggml.token_type"
  tokenAddBosKey = "tokenizer.ggml.add_bos_token"
  tokenAddEosKey = "tokenizer.ggml.add_eos_token"
  tokenAddPrefixKey = "tokenizer.ggml.add_space_prefix"
  tokenBosIdKey = "tokenizer.ggml.bos_token_id"
  tokenEosIdKey = "tokenizer.ggml.eos_token_id"
  tokenUnkIdKey = "tokenizer.ggml.unknown_token_id"
  tokenChatTemplateKey = "tokenizer.chat_template"

type
  TokenAttr* = enum
    tokNormal, tokUnknown, tokControl, tokUserDefined, tokUnused, tokByte

  TokenData* = object
    text*: string
    score*: float32
    attr*: TokenAttr

  Vocab* = object
    tokens*: seq[TokenData]
    tokenToId*: Table[string, int]
    addBos*: bool
    addEos*: bool
    addSpacePrefix*: bool
    bosId*: int32
    eosId*: int32
    unkId*: int32
    modelType*: string
    chatTemplate*: string
    bpeRanks*: Table[string, int]

  Symbol = object
    prev, next: int
    start, len: int

  Bigram = object
    keyScore: float32
    keyLeft: int
    left, right: int
    size: int

proc `<`(a, b: Bigram): bool =
  if a.keyScore == b.keyScore:
    return a.keyLeft < b.keyLeft
  a.keyScore < b.keyScore

var
  gpt2MapsInited = false
  gpt2ByteEncode: array[256, string]
  gpt2ByteDecode: Table[int32, uint8]

proc initGpt2ByteMaps() =
  if gpt2MapsInited:
    return
  var bs: seq[int] = @[]
  for i in 33 .. 126: bs.add(i)
  for i in 161 .. 172: bs.add(i)
  for i in 174 .. 255: bs.add(i)
  var present = newSeq[bool](256)
  for b in bs:
    present[b] = true
  var cs = bs
  var n = 0
  for b in 0 .. 255:
    if not present[b]:
      bs.add(b)
      cs.add(256 + n)
      inc n
  gpt2ByteDecode = initTable[int32, uint8](256)
  for i in 0 ..< bs.len:
    let b = bs[i]
    let cp = cs[i]
    let piece = $Rune(cp)
    gpt2ByteEncode[b] = piece
    gpt2ByteDecode[int32(cp)] = uint8(b)
  gpt2MapsInited = true

proc utf8Len(b: byte): int =
  if b < 0x80: return 1
  if b < 0xE0: return 2
  if b < 0xF0: return 3
  if b < 0xF8: return 4
  1

proc escapeWhitespace(s: string): string =
  result = s.replace(" ", "\xE2\x96\x81")

proc hexByte(ch: byte): string =
  const hex = "0123456789ABCDEF"
  result = "<0x" & $hex[int(ch) shr 4] & $hex[int(ch) and 15] & ">"

proc byteToToken(v: Vocab, ch: byte): int32 =
  let t1 = hexByte(ch)
  if v.tokenToId.hasKey(t1):
    return int32(v.tokenToId[t1])
  let t2 = $char(ch)
  if v.tokenToId.hasKey(t2):
    return int32(v.tokenToId[t2])
  v.unkId

proc loadVocab*(g: GgufFile): Vocab =
  var modelType: string
  discard g.getKvStr(tokenModelKey, modelType)
  var tokenList: seq[string]
  let okTokens = g.getKvArrStr(tokenListKey, tokenList)
  if not okTokens:
    raise newException(IOError, "tokenizer tokens missing")

  var scores: seq[float32]
  discard g.getKvArrF32(tokenScoresKey, scores)
  if scores.len != tokenList.len:
    scores.setLen(tokenList.len)

  var tokenTypes: seq[int32]
  discard g.getKvArrI32(tokenTypesKey, tokenTypes)
  if tokenTypes.len != tokenList.len:
    tokenTypes.setLen(tokenList.len)
  var merges: seq[string]
  discard g.getKvArrStr("tokenizer.ggml.merges", merges)

  var addBos = false
  var addEos = false
  var addSpacePrefix = true
  let hasAddBos = g.getKvBool(tokenAddBosKey, addBos)
  discard g.getKvBool(tokenAddEosKey, addEos)
  let hasAddPrefix = g.getKvBool(tokenAddPrefixKey, addSpacePrefix)
  if modelType == "gpt2" and not hasAddPrefix:
    addSpacePrefix = false
  if modelType == "llama" and not hasAddBos:
    addBos = true

  var bosId: int32 = 1
  var eosId: int32 = 2
  var unkId: int32 = 0
  proc getTokenId(g: GgufFile, key: string, value: var int32) =
    # token ids are written as u32 by most converters, as i32 by some
    if not g.getKvI32(key, value):
      var u: uint32
      if g.getKvU32(key, u):
        value = int32(u)
  getTokenId(g, tokenBosIdKey, bosId)
  getTokenId(g, tokenEosIdKey, eosId)
  getTokenId(g, tokenUnkIdKey, unkId)

  result.modelType = modelType
  result.addBos = addBos
  result.addEos = addEos
  result.addSpacePrefix = addSpacePrefix
  result.bosId = bosId
  result.eosId = eosId
  result.unkId = unkId
  discard g.getKvStr(tokenChatTemplateKey, result.chatTemplate)

  result.tokens.setLen(tokenList.len)
  for i, tok in tokenList:
    result.tokens[i].text = tok
    result.tokens[i].score = scores[i]
    result.tokens[i].attr = tokNormal

  result.tokenToId = initTable[string, int](tokenList.len * 2)
  for i, tok in tokenList:
    result.tokenToId[tok] = i
  result.bpeRanks = initTable[string, int](max(merges.len * 2, 16))
  for i, m in merges:
    let sp = m.find(' ')
    if sp > 0 and sp + 1 < m.len:
      let a = m[0 ..< sp]
      let b = m[sp + 1 .. ^1]
      result.bpeRanks[a & "\t" & b] = i

proc tokenizeSpm(v: Vocab, text: string): seq[int32] =
  var raw = text
  if v.addSpacePrefix and raw.len > 0:
    raw = " " & raw
  raw = escapeWhitespace(raw)

  var symbols: seq[Symbol] = @[]
  var revMerge = initTable[string, (int, int)]()
  var work = initHeapQueue[Bigram]()

  var index = 0
  var offs = 0
  while offs < raw.len:
    let b = raw[offs].byte
    let n = min(utf8Len(b), raw.len - offs)
    var sym: Symbol
    sym.start = offs
    sym.len = n
    sym.prev = index - 1
    sym.next = (if offs + n == raw.len: -1 else: index + 1)
    symbols.add(sym)
    offs += n
    inc index

  proc tryAddBigram(left, right: int) =
    if left < 0 or right < 0: return
    let a = symbols[left]
    let b = symbols[right]
    if a.len == 0 or b.len == 0: return
    let textLen = a.len + b.len
    let text = raw.substr(a.start, a.start + textLen - 1)
    if not v.tokenToId.hasKey(text): return
    let id = v.tokenToId[text]
    if id < 0 or id >= v.tokens.len: return
    let score = v.tokens[id].score
    work.push(Bigram(
      keyScore: -score,
      keyLeft: left,
      left: left,
      right: right,
      size: textLen
    ))
    revMerge[text] = (left, right)

  for i in 1 ..< symbols.len:
    tryAddBigram(i - 1, i)

  while work.len > 0:
    let bigram = work.pop()
    var leftSym = symbols[bigram.left]
    var rightSym = symbols[bigram.right]
    if leftSym.len == 0 or rightSym.len == 0: continue
    if leftSym.len + rightSym.len != bigram.size: continue

    leftSym.len += rightSym.len
    rightSym.len = 0
    leftSym.next = rightSym.next
    if rightSym.next >= 0:
      symbols[rightSym.next].prev = bigram.left
    symbols[bigram.left] = leftSym
    symbols[bigram.right] = rightSym

    tryAddBigram(leftSym.prev, bigram.left)
    tryAddBigram(bigram.left, leftSym.next)

  proc resegment(idx: int, outp: var seq[int32]) =
    let sym = symbols[idx]
    if sym.len == 0: return
    let text = raw.substr(sym.start, sym.start + sym.len - 1)
    if v.tokenToId.hasKey(text):
      outp.add(int32(v.tokenToId[text]))
      return
    if revMerge.hasKey(text):
      let (l, r) = revMerge[text]
      resegment(l, outp)
      resegment(r, outp)
      return
    for i in 0 ..< sym.len:
      outp.add(byteToToken(v, raw[sym.start + i].byte))

  var outp: seq[int32] = @[]
  var i = 0
  while i >= 0 and i < symbols.len:
    if symbols[i].prev == -1 or i == 0:
      break
    i = symbols[i].prev
  for j in 0 ..< symbols.len:
    if symbols[j].prev == -1:
      i = j
      break
  while i != -1:
    resegment(i, outp)
    i = symbols[i].next

  outp

proc tokenizeGpt2Bytes(v: Vocab, text: string): seq[int32] =
  ## GPT2/Qwen2 byte-level encode with merge-rank BPE.
  initGpt2ByteMaps()
  var raw = text
  if v.addSpacePrefix and raw.len > 0:
    raw = " " & raw
  var symbols: seq[string] = @[]
  symbols.setLen(raw.len)
  for i, ch in raw:
    let b = ord(ch)
    if b < 0 or b > 255:
      symbols[i] = ""
    else:
      symbols[i] = gpt2ByteEncode[b]

  if v.bpeRanks.len > 0 and symbols.len >= 2:
    while true:
      var bestRank = high(int)
      var bestPos = -1
      for i in 0 ..< symbols.len - 1:
        let key = symbols[i] & "\t" & symbols[i + 1]
        if v.bpeRanks.hasKey(key):
          let r = v.bpeRanks[key]
          if r < bestRank:
            bestRank = r
            bestPos = i
      if bestPos < 0:
        break
      symbols[bestPos] = symbols[bestPos] & symbols[bestPos + 1]
      symbols.delete(bestPos + 1)

  result = @[]
  for piece in symbols:
    if piece.len == 0:
      result.add(v.unkId)
    elif v.tokenToId.hasKey(piece):
      result.add(int32(v.tokenToId[piece]))
    else:
      var matched = false
      for r in piece.runes:
        let rs = $r
        if v.tokenToId.hasKey(rs):
          result.add(int32(v.tokenToId[rs]))
          matched = true
        else:
          result.add(v.unkId)
      if not matched:
        result.add(v.unkId)

proc tokenize*(v: Vocab, text: string, addSpecial = true): seq[int32] =
  if v.modelType == "gpt2":
    result = tokenizeGpt2Bytes(v, text)
  else:
    result = tokenizeSpm(v, text)
  if addSpecial and v.addBos:
    result.insert(v.bosId, 0)
  if addSpecial and v.addEos:
    result.add(v.eosId)

proc tokenizeSegment(v: Vocab, text: string): seq[int32] =
  if v.modelType == "gpt2":
    tokenizeGpt2Bytes(v, text)
  else:
    tokenizeSpm(v, text)

proc tokenizeWithSpecial*(v: Vocab, text: string, addSpecial = true): seq[int32] =
  var specials = @[
    "<|user|>", "<|assistant|>", "<|system|>",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "</s>", "<s>"
  ]
  specials = specials.filterIt(v.tokenToId.hasKey(it))
  if specials.len == 0:
    return tokenize(v, text, addSpecial)

  var pos = 0
  var outTokens: seq[int32] = @[]
  while pos < text.len:
    var bestIdx = -1
    var bestTok = ""
    for s in specials:
      let i = text.find(s, pos)
      if i >= 0 and (bestIdx == -1 or i < bestIdx):
        bestIdx = i
        bestTok = s
    if bestIdx == -1:
      outTokens.add(tokenizeSegment(v, text.substr(pos)))
      break
    if bestIdx > pos:
      outTokens.add(tokenizeSegment(v, text.substr(pos, bestIdx - 1)))
    outTokens.add(int32(v.tokenToId[bestTok]))
    pos = bestIdx + bestTok.len

  if addSpecial and v.addBos:
    outTokens.insert(v.bosId, 0)
  if addSpecial and v.addEos:
    outTokens.add(v.eosId)
  outTokens

proc tokenToPiece*(v: Vocab, id: int32): string =
  let idx = int(id)
  if idx < 0 or idx >= v.tokens.len:
    return ""
  result = v.tokens[idx].text

proc detokenize*(v: Vocab, tokens: seq[int32]): string =
  if v.modelType == "gpt2":
    initGpt2ByteMaps()
    var outBytes: seq[byte] = @[]
    for t in tokens:
      let piece = tokenToPiece(v, t)
      if t == v.bosId or t == v.eosId:
        continue
      if piece == "<s>" or piece == "</s>" or piece == "<|user|>" or
         piece == "<|assistant|>" or piece == "<|system|>" or
         piece == "<|im_start|>" or piece == "<|im_end|>" or
         piece == "<|endoftext|>":
        continue
      for r in piece.runes:
        let ri = int32(r)
        if gpt2ByteDecode.hasKey(ri):
          outBytes.add(gpt2ByteDecode[ri])
        else:
          let utf = $r
          for c in utf:
            outBytes.add(byte(ord(c)))
    result = newString(outBytes.len)
    for i, b in outBytes:
      result[i] = char(b)
    return

  result = ""
  for t in tokens:
    let piece = tokenToPiece(v, t)
    if t == v.bosId or t == v.eosId:
      continue
    if piece == "<s>" or piece == "</s>" or piece == "<|user|>" or
       piece == "<|assistant|>" or piece == "<|system|>":
      continue
    if piece.len == 6 and piece.startsWith("<0x") and piece.endsWith(">"):
      let hex = piece[3..4]
      try:
        let b = parseHexInt(hex)
        result.add(char(b))
      except ValueError:
        result.add(piece)
    else:
      result.add(piece)
  result = result.replace("\xE2\x96\x81", " ")

proc formatChatPrompt*(v: Vocab, userText: string): string =
  ## Minimal support for TinyLlama and Qwen-style chat templates.
  if v.chatTemplate.len == 0:
    return userText
  if v.chatTemplate.contains("<|im_start|>") and v.chatTemplate.contains("<|im_end|>"):
    return "<|im_start|>user\n" & userText & "<|im_end|>\n<|im_start|>assistant\n"
  if v.chatTemplate.contains("<|user|>") and v.chatTemplate.contains("<|assistant|>"):
    let eosPiece = tokenToPiece(v, v.eosId)
    return "<|user|>\n" & userText & eosPiece & "<|assistant|>"
  userText
