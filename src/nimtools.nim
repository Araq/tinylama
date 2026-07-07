## Agent tool registry: tools are argv templates with $var substitution,
## so external tooling (valgrind, linters, ...) plugs in via config.
## The built-in tools wrap nimony's one-shot IDE commands
## (`nimony check --def/--usages`). Tool calls are parsed from Hermes-style
## <tool_call> blocks as emitted by Qwen instruct models.

import std/[json, monotimes, os, osproc, strutils, tables, times]
when defined(windows):
  import std/streams
else:
  from std/posix import read, fcntl, F_GETFL, F_SETFL, O_NONBLOCK,
    EAGAIN, EWOULDBLOCK, EINTR, errno

const maxToolOutput = 4096 ## keep tool results small; they feed a context window

type
  ToolParam* = object
    name*: string
    typ*: string          ## JSON schema type: "string", "integer", ...
    description*: string

  ToolDef* = object
    name*: string
    description*: string
    params*: seq[ToolParam]
    exec*: seq[string]    ## argv template; elements may contain $param vars
    workDir*: string      ## working directory template; "" means current dir
    native*: bool         ## implemented in-process (file tools), no exec

  ToolCall* = object
    name*: string
    arguments*: JsonNode

proc param(name, typ, description: string): ToolParam =
  ToolParam(name: name, typ: typ, description: description)

proc builtinTools*(): seq[ToolDef] =
  ## nimony resolves positions against line info that uses paths relative
  ## to the project directory, hence $fileName + workDir $fileDir.
  let posParams = @[
    param("file", "string", "Path to the .nim file"),
    param("line", "integer", "1-based line of the symbol"),
    param("col", "integer", "1-based column of the symbol")]
  @[
    ToolDef(name: "goto_def",
      description: "Find where the symbol at the given source position is " &
        "defined in a Nim project. Returns the definition location as: " &
        "def <symbol> <file> <line> <col>.",
      params: posParams,
      exec: @["$nimony", "check", "--silentMake",
              "--def:$fileName,$line,$col", "$fileName"],
      workDir: "$fileDir"),
    ToolDef(name: "find_usages",
      description: "List all usages of the symbol at the given source " &
        "position in a Nim project. Returns one line per usage as: " &
        "use <symbol> <file> <line> <col>.",
      params: posParams,
      exec: @["$nimony", "check", "--silentMake",
              "--usages:$fileName,$line,$col", "$fileName"],
      workDir: "$fileDir"),
    ToolDef(name: "write_file",
      description: "Create or overwrite a file with the given content. " &
        "Paths are relative to the working directory. For long files, " &
        "write the first part with write_file and continue with append_file.",
      params: @[
        param("path", "string", "Relative path of the file to write"),
        param("content", "string", "The complete new content of the file")],
      native: true),
    ToolDef(name: "append_file",
      description: "Append content to the end of a file, creating it if " &
        "missing. Use this to build up long files chunk by chunk.",
      params: @[
        param("path", "string", "Relative path of the file"),
        param("content", "string", "Content to append at the end")],
      native: true),
    ToolDef(name: "read_file",
      description: "Read a file and return its content.",
      params: @[
        param("path", "string", "Relative path of the file to read")],
      native: true)
  ]

proc addOrReplace*(tools: var seq[ToolDef], t: ToolDef) =
  for existing in tools.mitems:
    if existing.name == t.name:
      existing = t
      return
  tools.add t

proc toolSpec(t: ToolDef): JsonNode =
  var props = newJObject()
  var required = newJArray()
  for p in t.params:
    props[p.name] = %*{"type": p.typ, "description": p.description}
    required.add %p.name
  %*{"type": "function", "function": {
      "name": t.name, "description": t.description,
      "parameters": {"type": "object", "properties": props,
                     "required": required}}}

proc systemPrompt*(tools: seq[ToolDef]): string =
  var specs = ""
  for t in tools:
    specs.add $toolSpec(t) & "\n"
  result = """You are a helpful assistant specialized in the Nim programming language. You answer questions about Nim code precisely, using the provided tools to navigate the code when needed.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" & specs.strip() & """

</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

For example, to write a file you reply exactly:
<tool_call>
{"name": "write_file", "arguments": {"path": "example.nim", "content": "echo 42\n"}}
</tool_call>"""

proc toToolCall(j: JsonNode): ToolCall =
  ## Accepts the canonical {"name": ..., "arguments": {...}} plus the
  ## variants small models like to emit: a "tool_name" key, or the whole
  ## thing nested under a "tool_call"/"function" wrapper.
  var j = j
  for wrapper in ["tool_call", "function"]:
    if j.kind == JObject and j.hasKey(wrapper) and j[wrapper].kind == JObject:
      j = j[wrapper]
  if j.kind != JObject:
    return ToolCall(name: "", arguments: newJObject())
  result.name =
    if j.hasKey("name"): j["name"].getStr
    elif j.hasKey("tool_name"): j["tool_name"].getStr
    else: ""
  result.arguments =
    if j.hasKey("arguments") and j["arguments"].kind == JObject: j["arguments"]
    else: newJObject()

proc normalizeTripleQuotes(body: string): string =
  ## Coder models like to wrap string values in Python-style triple quotes,
  ## which is invalid JSON; rewrite them as escaped JSON strings.
  result = ""
  var pos = 0
  while true:
    let start = body.find("\"\"\"", pos)
    if start < 0:
      result.add body.substr(pos)
      break
    let stop = body.find("\"\"\"", start + 3)
    if stop < 0:
      result.add body.substr(pos)
      break
    result.add body.substr(pos, start - 1)
    result.add escapeJson(body[start + 3 ..< stop])
    pos = stop + 3

proc escapeRawControlChars(body: string): string =
  ## Models sometimes emit literal newlines or tabs inside JSON string
  ## literals (typically in "content" values), which strict JSON rejects;
  ## escape them so the call still parses.
  result = newStringOfCap(body.len + 16)
  var inStr = false
  var i = 0
  while i < body.len:
    let c = body[i]
    if inStr:
      case c
      of '\\':
        result.add c
        if i + 1 < body.len:
          inc i
          result.add body[i]
      of '"':
        inStr = false
        result.add c
      of '\n': result.add "\\n"
      of '\r': result.add "\\r"
      of '\t': result.add "\\t"
      of '\0'..'\x08', '\x0B', '\x0C', '\x0E'..'\x1F':
        result.add "\\u00" & toHex(ord(c), 2)
      else:
        result.add c
    else:
      if c == '"': inStr = true
      result.add c
    inc i

proc parseCallBody(body: string): JsonNode =
  try:
    return parseJson(body)
  except JsonParsingError, ValueError:
    discard
  let repaired = normalizeTripleQuotes(body)
  try:
    result = parseJson(repaired)
  except JsonParsingError, ValueError:
    result = parseJson(escapeRawControlChars(repaired))

proc parseToolCalls*(text: string, errors: var seq[string]): seq[ToolCall] =
  ## Extract <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks.
  ## Attempted-but-broken calls are appended to `errors` so the harness can
  ## feed the problem back to the model instead of silently dropping it.
  result = @[]
  var sawTag = false
  var pos = 0
  while true:
    let start = text.find("<tool_call>", pos)
    if start < 0: break
    sawTag = true
    let stop = text.find("</tool_call>", start)
    if stop < 0:
      errors.add "unterminated <tool_call> block (missing </tool_call>)"
      break
    let body = text[start + "<tool_call>".len ..< stop].strip()
    pos = stop + "</tool_call>".len
    try:
      let tc = toToolCall(parseCallBody(body))
      if tc.name.len > 0:
        result.add tc
      else:
        errors.add "tool call is missing a \"name\""
    except JsonParsingError, ValueError:
      errors.add "malformed tool call JSON: " & getCurrentExceptionMsg()
  if result.len == 0 and not sawTag:
    # Small models often emit the tool-call JSON but forget the
    # <tool_call> wrapper (or fence it in markdown); accept a bare
    # {"name"/"tool_name":..., "arguments":{...}} object.
    let start = text.find('{')
    if start >= 0:
      let stop = text.rfind('}')
      if stop > start:
        let body = text[start .. stop]
        # require an explicit "arguments" object so ordinary JSON answers
        # are not mistaken for tool calls
        if "\"arguments\"" in body:
          try:
            let tc = toToolCall(parseCallBody(body))
            if tc.name.len > 0:
              result.add tc
            else:
              errors.add "tool call is missing a \"name\""
          except JsonParsingError, ValueError:
            errors.add "malformed tool call JSON: " & getCurrentExceptionMsg()

proc parseToolCalls*(text: string): seq[ToolCall] =
  var errors: seq[string] = @[]
  parseToolCalls(text, errors)

proc subst(tpl: string, vars: Table[string, string]): string =
  result = ""
  var i = 0
  while i < tpl.len:
    if tpl[i] == '$' and i + 1 < tpl.len:
      var j = i + 1
      while j < tpl.len and tpl[j] in {'a'..'z', 'A'..'Z', '0'..'9', '_'}:
        inc j
      let key = tpl[i + 1 ..< j]
      if vars.hasKey(key):
        result.add vars[key]
        i = j
      else:
        result.add tpl[i]
        inc i
    else:
      result.add tpl[i]
      inc i

proc argToString(v: JsonNode): string =
  case v.kind
  of JString: v.getStr
  of JInt: $v.getInt
  of JFloat: $v.getFloat
  of JBool: $v.getBool
  else: $v

proc sandboxedPath(path: string): string =
  ## File tools may only touch files below the current working directory.
  if path.len == 0:
    raise newException(ValueError, "empty path")
  let root = getCurrentDir()
  result = absolutePath(path)
  if result != root and not result.startsWith(root & $DirSep):
    raise newException(ValueError,
      "path escapes the working directory: " & path)

proc execNativeTool(name: string, arguments: JsonNode): string =
  let path = sandboxedPath(arguments{"path"}.getStr)
  case name
  of "write_file":
    let content = arguments{"content"}.getStr
    createDir(parentDir(path))
    writeFile(path, content)
    result = "wrote " & $content.len & " bytes to " & relativePath(path, getCurrentDir())
  of "append_file":
    let content = arguments{"content"}.getStr
    createDir(parentDir(path))
    let f = open(path, fmAppend)
    f.write(content)
    f.close()
    result = "appended " & $content.len & " bytes to " &
      relativePath(path, getCurrentDir()) & " (now " & $getFileSize(path) & " bytes)"
  of "read_file":
    result = readFile(path)
    if result.len > maxToolOutput:
      result = result[0 ..< maxToolOutput] & "\n...(truncated)"
    elif result.len == 0:
      result = "(empty file)"
  else:
    result = "error: unknown native tool: " & name

const maxCapturedOutput = 65536 ## keep reading past this (so the child never
                                ## blocks on a full pipe) but stop storing

when not defined(windows):
  proc setNonBlocking(fd: cint) =
    let flags = fcntl(fd, F_GETFL, 0)
    if flags >= 0:
      discard fcntl(fd, F_SETFL, flags or O_NONBLOCK)

  proc drainPipe(p: Process, output: var string): bool =
    ## Append whatever is currently readable on p's stdout, which was set to
    ## O_NONBLOCK: a raw read returns what is available, EAGAIN when there is
    ## nothing yet, and 0 at EOF. (osproc.hasData is no help here — its posix
    ## version selects with no timeout, i.e. it blocks until data arrives.)
    ## Bounded per call so a spamming tool cannot starve the caller's
    ## timeout check. Returns false once EOF was reached.
    var buf: array[4096, char]
    var rounds = 64
    while rounds > 0:
      dec rounds
      let n = read(p.outputHandle.cint, addr buf[0], buf.len)
      if n == 0:
        return false # EOF: every write end is closed
      if n < 0:
        if errno == EINTR: continue
        return errno == EAGAIN or errno == EWOULDBLOCK # anything else is fatal
      if output.len < maxCapturedOutput:
        let old = output.len
        output.setLen old + n
        copyMem(addr output[old], addr buf[0], n)
    true

proc runToolProcess(argv: seq[string], workDir: string, timeoutSec: int):
    tuple[output: string, code: int, timedOut: bool] =
  ## Run argv directly, without a shell in between. Output is drained while
  ## the process runs so it cannot deadlock on a full pipe; when the deadline
  ## passes the process is terminated (then killed if it ignores that).
  var p = startProcess(argv[0], workDir, argv[1 .. ^1],
                       options = {poStdErrToStdOut, poUsePath})
  defer: p.close()
  result = (output: "", code: 0, timedOut: false)
  when defined(windows):
    # No non-blocking pipe reads here; fall back to a plain blocking run.
    result.output = p.outputStream.readAll()
  else:
    setNonBlocking(p.outputHandle.cint)
    let deadline = getMonoTime() + initDuration(seconds = timeoutSec)
    block ioLoop:
      while true:
        if not drainPipe(p, result.output): break ioLoop # EOF
        if not p.running:
          # Drain the leftovers until EOF; the grace deadline guards against
          # an orphaned grandchild keeping the pipe open but silent.
          let grace = getMonoTime() + initDuration(seconds = 2)
          while drainPipe(p, result.output):
            if getMonoTime() >= grace: break
            sleep(5)
          break ioLoop
        if timeoutSec > 0 and getMonoTime() >= deadline:
          result.timedOut = true
          p.terminate()
          for _ in 0 ..< 50:
            if not p.running: break
            sleep(20)
          if p.running: p.kill()
          discard drainPipe(p, result.output)
          break ioLoop
        sleep(20)
  result.code = p.waitForExit()

proc execTool*(tc: ToolCall, tools: seq[ToolDef], nimonyBin: string,
               timeoutSec = 60): string =
  var def: ToolDef
  var found = false
  for t in tools:
    if t.name == tc.name:
      def = t
      found = true
      break
  if not found:
    return "error: unknown tool: " & tc.name

  if def.native:
    try:
      return execNativeTool(tc.name, tc.arguments)
    except CatchableError as e:
      return "error: " & e.msg

  var vars = initTable[string, string]()
  vars["nimony"] = nimonyBin
  vars["cwd"] = getCurrentDir()
  for p in def.params:
    let v = tc.arguments{p.name}
    vars[p.name] = if v.isNil: "" else: argToString(v)
  if vars.hasKey("file") and vars["file"].len > 0:
    let absFile = absolutePath(vars["file"])
    if not fileExists(absFile):
      return "error: no such file: " & vars["file"]
    vars["file"] = absFile
    vars["fileName"] = extractFilename(absFile)
    vars["fileDir"] = parentDir(absFile)

  var argv: seq[string] = @[]
  for tpl in def.exec:
    let a = subst(tpl, vars)
    if a.len > 0: argv.add a
  if argv.len == 0:
    return "error: tool " & tc.name & " has an empty command"
  var workDir = subst(def.workDir, vars)
  if workDir.len == 0: workDir = getCurrentDir()

  var toolRes: tuple[output: string, code: int, timedOut: bool]
  try:
    toolRes = runToolProcess(argv, workDir, timeoutSec)
  except OSError as e:
    return "error running " & tc.name & ": " & e.msg
  var res = toolRes.output.strip()
  if res.len > maxToolOutput:
    res = res[0 ..< maxToolOutput] & "\n...(output truncated)"
  if toolRes.timedOut:
    return "error: " & tc.name & " timed out after " & $timeoutSec &
      "s and was killed" & (if res.len > 0: ":\n" & res else: "")
  if toolRes.code != 0:
    return "error running " & tc.name & " (exit " & $toolRes.code & "):\n" & res
  if res.len == 0:
    return "no results"
  res
