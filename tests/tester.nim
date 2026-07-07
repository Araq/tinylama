import std/os

proc fatal(msg: string) = quit "FAILURE " & msg

proc exec(cmd: string) =
  if execShellCmd(cmd) != 0: fatal cmd

exec "nim c src/tinylama.nim"
exec "nim c src/harness.nim"
exec "nim c -r tests/tharness.nim"
# SIMPLE (CPU) runtime exercises the rope/bias kernels; the warp kernels
# (WarpSize==32, incl. Q8_0 GEMV/GEMM) are skipped here and validated on a
# real GPU via: nim cpp -r --cc:hipcc -d:useHippo tests/thippo_kernels.nim
exec "nim cpp -r -d:useHippo -d:HippoRuntime=SIMPLE -d:useMalloc tests/thippo_kernels.nim"