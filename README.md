# tinylama

Tiny Nim prototype that loads GGUF models and runs a minimal LLaMA-style
forward pass with greedy decoding.

## Build

```bash
nim c -r src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf "hello" --max-new 16
```

Optional progress output:

```bash
nim c -r src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf "hello" --max-new 16 --progress
```

Optional Malebolgia parallelization (requires Malebolgia available to Nim):

```bash
nim c -r -d:useMalebolgia -d:ThreadPoolSize=8 -d:FixedChanSize=16 \
  --path:/home/araq/projects/malebolgia/src \
  src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf "hello" --max-new 16
```

Optional Hippo backend (HIP via hipcc, AMD):

```bash
HIP_PLATFORM=amd nim cpp -r -d:release --cc:hipcc \
  -d:useHippo -d:useMalloc --path:../hippo/src \
  src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf "hello" --max-new 16
```

Optional Hippo backend (CUDA via nvcc, NVIDIA):

```bash
NVCC_PREPEND_FLAGS="-arch=sm_86" nim cpp -r -d:release --cc:nvcc \
  -d:useHippo -d:HippoRuntime=CUDA -d:useMalloc --path:../hippo/src \
  src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf "hello" --max-new 16
```

This command was validated on AWS `g5.xlarge` (NVIDIA A10G, CUDA 13.1 toolkit).
The `NVCC_PREPEND_FLAGS="-arch=sm_86"` setting avoids a PTX/runtime mismatch on this GPU.

## Download the tested model

This project was tested with the TinyLlama 1.1B Q2_K GGUF.

```bash
mkdir -p models
curl -L -o models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
```

## Agent harness (Nim-dedicated)

`src/harness.nim` is an agentic loop on top of the inference core: it drives a
Qwen-style instruct model (ChatML, Hermes tool calls) and gives it tools to
navigate Nim code via [nimony](https://github.com/nim-lang/nimony)'s one-shot
IDE commands (`nimony check --def/--usages`).

Requirements: a `nimony` checkout as a sibling of this repository (built
binaries in `nimony/bin/`; the NIF libraries are imported from
`../nimony/src/lib`, see `nim.cfg`).

```bash
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q8_0.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

nim c -r -d:release src/harness.nim models/qwen2.5-0.5b-instruct-q8_0.gguf \
  "Call the find_usages function with file=\"tests/demo.nim\", line=3, col=9, then report the locations."
```

Options: `--max-new N` (tokens per round, default 512), `--max-continue N`
(rounds a single turn may continue past the token budget, default 4 —
long generations resume seamlessly instead of being truncated), `--temp T`
(default 0.7, 0 = greedy), `--top-p P` (default 0.9), `--repeat-penalty P`
(default 1.1, 1.0 = off), `--repeat-window N` (default 256), `--seed N`,
`--ctx N` (KV cache size, default 4096), `--max-turns N` (tool-use rounds,
default 8), `--nimony PATH` (or `NIMONY` env var), `--tools FILE.nif`,
`--verbose` (dump prompts).

### Configurable tools

The built-in tools are `goto_def`, `find_usages`, and the native file tools
`write_file`, `append_file`, `read_file` — the model produces long files
chunk by chunk on disk instead of inside its context window. File tools are
sandboxed to paths below the harness's working directory. Additional tools are
loaded from a NIF file (`harness.tools.nif` in the current directory, or
`--tools FILE.nif`); see the checked-in `harness.tools.nif` for `nim_check`
and `valgrind_memcheck` examples. Each tool is an argv template with
`$param` substitution:

```
(tool "nim_check"
 (description "Type-check a Nim file and report errors.")
 (param "file" "string" "Path to the .nim file")
 (exec "nim" "check" "--hints:off" "$fileName")
 (dir "$fileDir"))
```

Besides the declared params, templates may use `$nimony` (path to the nimony
binary), `$cwd`, and — when a `file` param is present — `$fileName` and
`$fileDir`. Tool output is truncated to 4 KB before it is fed back to the
model.

## Notes

- The default forward pass is CPU and naive (no batching, no optimizations).
- The Hippo backend supports qwen2-style models (NEOX rope, QKV biases).
  Its rope/bias kernels can be tested without a GPU via hippo's pure-CPU runtime:
  `nim cpp -r -d:useHippo -d:HippoRuntime=SIMPLE -d:useMalloc tests/thippo_kernels.nim`
  The warp kernels (incl. the Q8_0 GEMV/GEMM) need a real GPU; validate them with
  `HIP_PLATFORM=amd nim cpp -r --cc:hipcc -d:useHippo -d:useMalloc tests/thippo_kernels.nim`
- Q8_0 weights stay quantized on the GPU (dequantized on the fly in the GEMV/GEMM
  kernels), so a q8_0 model uses ~1/4 the weight memory of the f32-upload path and
  decodes faster (less memory traffic). This is what makes 7B-class models fit.
- KV cache is enabled for decode steps to improve speed.
- Only GGUF models with LLaMA architecture and supported quant types
  (Q2_K/Q3_K/Q6_K/F16/F32) are currently supported.

## Benchmarking with Benchy

Install bench dependency:

```bash
nimble install -y benchy
```

Run benchmarks in release mode:

```bash
nim c -r -d:release bench/bench_tinylama.nim \
  models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
```

The benchmark prints decode throughput as `avg tok/s ± stdev` (matching llama-bench).
Defaults: 128 decode steps, 1 warmup run, 5 sample runs.
Override with `--decode-steps N`, `--decode-warmup N`, `--decode-runs N`.

Optional Malebolgia parallel run:

```bash
nim c -r -d:release -d:useMalebolgia -d:ThreadPoolSize=8 -d:FixedChanSize=16 \
  bench/bench_tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
```

Optional Hippo benchmark run (HIP via hipcc, AMD):

```bash
HIP_PLATFORM=amd nim cpp -r -d:release --cc:hipcc \
  -d:useHippo -d:useMalloc --path:../hippo/src \
  bench/bench_tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
```

Optional Hippo benchmark run (CUDA via nvcc, NVIDIA):

```bash
NVCC_PREPEND_FLAGS="-arch=sm_86" nim cpp -r -d:release --cc:nvcc \
  -d:useHippo -d:HippoRuntime=CUDA -d:useMalloc --path:../hippo/src \
  bench/bench_tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
```

## Example

```bash
nim c -r src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf \
  "what is the capital of France?" --max-new 32
```

Example output:

```
The capital of France is Paris.
```
