## ChatML (Qwen-style) multi-turn prompt construction and stop tokens.

import std/tables
import ./tokenizer

type
  Role* = enum
    roleSystem = "system"
    roleUser = "user"
    roleAssistant = "assistant"
    roleTool = "tool"

  ChatMessage* = object
    role*: Role
    content*: string

proc msg*(role: Role, content: string): ChatMessage =
  ChatMessage(role: role, content: content)

proc renderChatML*(messages: openArray[ChatMessage],
                   addGenerationPrompt = true): string =
  ## Renders messages the way Qwen's chat template does: tool results are
  ## wrapped in <tool_response> tags inside a user turn; consecutive tool
  ## messages share one user turn.
  result = ""
  var i = 0
  while i < messages.len:
    let m = messages[i]
    case m.role
    of roleTool:
      result.add "<|im_start|>user"
      while i < messages.len and messages[i].role == roleTool:
        result.add "\n<tool_response>\n" & messages[i].content & "\n</tool_response>"
        inc i
      result.add "<|im_end|>\n"
    else:
      result.add "<|im_start|>" & $m.role & "\n" & m.content & "<|im_end|>\n"
      inc i
  if addGenerationPrompt:
    result.add "<|im_start|>assistant\n"

proc stopTokenIds*(v: Vocab): seq[int32] =
  ## Token ids that end an assistant turn.
  result = @[v.eosId]
  for piece in ["<|im_end|>", "<|endoftext|>", "</s>"]:
    if v.tokenToId.hasKey(piece):
      let id = int32(v.tokenToId[piece])
      if id notin result:
        result.add id
