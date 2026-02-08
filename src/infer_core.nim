## Shared inference helpers for CLI and benchmarks.

import ./tokenizer
import ./tensor

proc argmaxLast*(logits: Tensor, nVocab: int): int32 =
  ## Return the best token id from the final logits position.
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

proc encodePromptTokens*(vocab: Vocab, userText: string): seq[int32] =
  ## Format chat input and tokenize it with special token handling.
  let fullPrompt = formatChatPrompt(vocab, userText)
  let useSpecial = fullPrompt != userText
  result = tokenizeWithSpecial(vocab, fullPrompt, addSpecial = not useSpecial)
