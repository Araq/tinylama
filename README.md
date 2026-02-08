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

## Notes

- The default forward pass is CPU and naive (no batching, no optimizations).
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

Optional Malebolgia parallel run:

```bash
nim c -r -d:release -d:useMalebolgia -d:ThreadPoolSize=8 -d:FixedChanSize=16 \
  bench/bench_tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
```

Optional Hippo benchmark run (HIP via hipcc, AMD):

```bash
nim cpp -r -d:release --cc:hipcc \
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
