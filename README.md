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

Optional GPU acceleration (requires CUDA or OpenCL available to Nim and Arraymancer):

```bash
# For CUDA
nim c -r -d:cuda src/tinylama.nim models/model.gguf "hello"

# For Metal/Vulkan (via OpenCL)
nim c -r -d:opencl src/tinylama.nim models/model.gguf "hello"
```

Optional Malebolgia parallelization:

```bash
nim c -r -d:useMalebolgia src/tinylama.nim models/model.gguf "hello"
```

## Download the tested model

This project was tested with the TinyLlama 1.1B Q2_K GGUF.

```bash
mkdir -p models
curl -L -o models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
```

## Notes

- The current forward pass is CPU-only and naive (no batching, no optimizations).
- KV cache is enabled for decode steps to improve speed.
- Only GGUF models with LLaMA architecture and supported quant types
  (Q2_K/Q3_K/Q6_K/F16/F32) are currently supported.

## Example

```bash
nim c -r src/tinylama.nim models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf \
  "what is the capital of France?" --max-new 32
```

Example output:

```
The capital of France is Paris.
```
