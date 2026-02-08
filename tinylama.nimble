version       = "0.1.0"
author        = "araq"
description   = "Tiny Nim GGUF LLaMA runner"
license       = "MIT"

srcDir = "src"
bin = @["tinylama"]

requires "malebolgia >= 1.0.0"
requires "hippo >= 0.9.0"
requires "benchy >= 0.0.1"
