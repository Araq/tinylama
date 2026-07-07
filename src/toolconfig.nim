## Loads extra harness tools from a NIF file (dialect "harness-tools"),
## using nimony's NIF libraries directly (expects a nimony/ checkout as a
## sibling of this repository; see src/nim.cfg).
##
## Example tools file:
##
##   (.nif27)
##   (.vendor "tinylama")
##   (.dialect "harness-tools")
##   (stmts
##    (tool "valgrind_memcheck"
##     (description "Run valgrind's memcheck on an executable.")
##     (param "binary" "string" "Path to the executable")
##     (param "args" "string" "Command line arguments, may be empty")
##     (exec "valgrind" "--leak-check=summary" "$binary" "$args")
##     (dir "$cwd")))

include nifprelude
import ./nimtools

proc stringChildren(c: var Cursor): seq[string] =
  ## c sits on a ParLe; collects all string-literal children and
  ## consumes the whole subtree.
  result = @[]
  inc c
  while c.kind != ParRi:
    if c.kind == StringLit:
      result.add pool.strings[c.litId]
      inc c
    else:
      skip c
  inc c

proc parseTool(c: var Cursor): ToolDef =
  inc c # past (tool
  if c.kind == StringLit:
    result.name = pool.strings[c.litId]
    inc c
  while c.kind != ParRi:
    if c.kind == ParLe:
      let section = pool.tags[c.tag]
      case section
      of "description":
        let s = stringChildren(c)
        if s.len > 0: result.description = s[0]
      of "param":
        let s = stringChildren(c)
        if s.len >= 3:
          result.params.add ToolParam(name: s[0], typ: s[1], description: s[2])
      of "exec":
        result.exec = stringChildren(c)
      of "dir":
        let s = stringChildren(c)
        if s.len > 0: result.workDir = s[0]
      else:
        skip c
    else:
      inc c
  inc c # past ParRi

proc loadToolConfig*(path: string): seq[ToolDef] =
  ## Parses a harness-tools NIF file; raises IOError on malformed input.
  result = @[]
  var f = nifstreams.open(path)
  try:
    discard processDirectives(f.r)
    var buf = fromStream(f)
    var c = beginRead(buf)
    if c.kind != ParLe or pool.tags[c.tag] != "stmts":
      raise newException(IOError, path & ": expected (stmts ...) toplevel")
    inc c
    while c.kind != ParRi:
      if c.kind == ParLe and pool.tags[c.tag] == "tool":
        let t = parseTool(c)
        if t.name.len == 0 or t.exec.len == 0:
          raise newException(IOError, path & ": tool needs a name and an (exec ...)")
        result.add t
      else:
        skip c
  finally:
    f.close()
