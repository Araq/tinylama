## Minimal tokenizer for GGUF models (SPM/LLaMA style).

import std/[tables, heapqueue, strutils]
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

  var addBos = false
  var addEos = false
  var addSpacePrefix = true
  discard g.getKvBool(tokenAddBosKey, addBos)
  discard g.getKvBool(tokenAddEosKey, addEos)
  discard g.getKvBool(tokenAddPrefixKey, addSpacePrefix)

  var bosId: int32 = 1
  var eosId: int32 = 2
  var unkId: int32 = 0
  discard g.getKvI32(tokenBosIdKey, bosId)
  discard g.getKvI32(tokenEosIdKey, eosId)
  discard g.getKvI32(tokenUnkIdKey, unkId)

  result.modelType = modelType
  result.addBos = addBos
  result.addEos = addEos
  result.addSpacePrefix = addSpacePrefix
  result.bosId = bosId
  result.eosId = eosId
  result.unkId = unkId

  result.tokens.setLen(tokenList.len)
  for i, tok in tokenList:
    result.tokens[i].text = tok
    result.tokens[i].score = scores[i]
    result.tokens[i].attr = tokNormal

  result.tokenToId = initTable[string, int](tokenList.len * 2)
  for i, tok in tokenList:
    result.tokenToId[tok] = i

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

proc tokenize*(v: Vocab, text: string, addSpecial = true): seq[int32] =
  result = tokenizeSpm(v, text)
  if addSpecial and v.addBos:
    result.insert(v.bosId, 0)
  if addSpecial and v.addEos:
    result.add(v.eosId)

proc tokenToPiece*(v: Vocab, id: int32): string =
  let idx = int(id)
  if idx < 0 or idx >= v.tokens.len:
    return ""
  result = v.tokens[idx].text

proc detokenize*(v: Vocab, tokens: seq[int32]): string =
  result = ""
  for t in tokens:
    result.add(tokenToPiece(v, t))
  result = result.replace("\xE2\x96\x81", " ")
