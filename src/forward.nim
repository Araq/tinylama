import arraymancer
import std/[math, tables]
import ./model
import ./tensor

type
  KvCache* = object
    k*, v*: seq[GGTensor]
    maxLen*: int
    curLen*: int
    nHeadKv*: int
    headDim*: int

proc initKvCache*(m: var Model, maxLen: int): KvCache =
  let hp = m.hparams
  result.maxLen = maxLen
  result.curLen = 0
  result.headDim = hp.headDim

  var kvDim = hp.nHeadKv * hp.headDim
  # Try to find actual kvDim from tensors
  let testName = "blk.0.attn_k.weight"
  if m.infos.hasKey(testName):
    kvDim = int(m.infos[testName].ne[1])

  result.nHeadKv = if result.headDim > 0: kvDim div result.headDim else: hp.nHeadKv

  result.k = newSeq[GGTensor](hp.nLayer)
  result.v = newSeq[GGTensor](hp.nLayer)
  for i in 0 ..< hp.nLayer:
    result.k[i].at = newTensor[float32](kvDim, maxLen).toDevice()
    result.v[i].at = newTensor[float32](kvDim, maxLen).toDevice()


proc applyRopeInternal(x: var GGTensor, nHead: int, headDim: int, ropeDim: int, base: float32, posOffset: int, arch: string) =
  var cpuX = x.at.toCpu()
  let nTokens = cpuX.shape[1]
  let rDim = if ropeDim > 0: ropeDim else: headDim

  for p in 0 ..< nTokens:
    let pos = float32(p + posOffset)
    for i in 0 ..< rDim div 2:
      let theta = pow(base, -float32(2 * i) / float32(rDim))
      let cos_val = cos(pos * theta)
      let sin_val = sin(pos * theta)
      for h in 0 ..< nHead:
        if arch == "llama" or arch == "mistral" or arch == "gemma" or arch == "gemma2":
          # Split half RoPE
          let idx0 = h * headDim + i
          let idx1 = h * headDim + i + headDim div 2
          let v0 = cpuX[idx0, p]
          let v1 = cpuX[idx1, p]
          cpuX[idx0, p] = v0 * cos_val - v1 * sin_val
          cpuX[idx1, p] = v0 * sin_val + v1 * cos_val
        else:
          # Interleaved RoPE
          let idx0 = h * headDim + 2 * i
          let idx1 = h * headDim + 2 * i + 1
          let v0 = cpuX[idx0, p]
          let v1 = cpuX[idx1, p]
          cpuX[idx0, p] = v0 * cos_val - v1 * sin_val
          cpuX[idx1, p] = v0 * sin_val + v1 * cos_val
  x.at = cpuX.toDevice()

proc softmaxCols(x: var Tensor[float32]) =
  for j in 0 ..< x.shape[1]:
    var maxVal = x[0, j]
    for i in 1 ..< x.shape[0]:
      if x[i, j] > maxVal: maxVal = x[i, j]
    var sum = 0.0f
    for i in 0 ..< x.shape[0]:
      x[i, j] = exp(x[i, j] - maxVal)
      sum += x[i, j]
    for i in 0 ..< x.shape[0]:
      x[i, j] /= sum

proc forwardPrefill*(m: var Model, cache: var KvCache, tokens: seq[int32]): GGTensor =
  let hp = m.hparams
  var intTokens = newSeq[int](tokens.len)
  for i in 0 ..< tokens.len: intTokens[i] = int(tokens[i])
  var x = embeddingLookup(m.getTensor("token_embd.weight"), intTokens)

  let headDim = hp.headDim
  let ropeDim = hp.ropeDim

  for layer in 0 ..< hp.nLayer:
    let prefix = "blk." & $layer & "."
    let norm = m.getTensor(prefix & "attn_norm.weight")
    let wq = m.getTensor(prefix & "attn_q.weight")
    let wk = m.getTensor(prefix & "attn_k.weight")
    let wv = m.getTensor(prefix & "attn_v.weight")
    let wo = m.getTensor(prefix & "attn_output.weight")

    let xNorm = rmsnormCols(x, norm, hp.rmsEps, hp.normType == "gemma")

    var q = amMatmul(wq, xNorm)
    var k = amMatmul(wk, xNorm)
    var v = amMatmul(wv, xNorm)

    applyRopeInternal(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, 0, hp.arch)
    applyRopeInternal(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, 0, hp.arch)

    # Store in cache
    cache.k[layer].at[_, 0 ..< tokens.len] = k.at
    cache.v[layer].at[_, 0 ..< tokens.len] = v.at

    # Attention
    let cpuQ = q.at.toCpu()
    let cpuK = k.at.toCpu()
    let cpuV = v.at.toCpu()
    var attnOut = newTensor[float32](hp.nEmb, tokens.len)

    let nHead = hp.nHead
    let nHeadKv = hp.nHeadKv
    let nGroup = nHead div nHeadKv
    let scale = 1.0f / sqrt(float32(headDim))

    for h in 0 ..< nHead:
      let hkv = h div nGroup
      for t in 0 ..< tokens.len:
        var scores = newTensor[float32](t + 1, 1)
        for i in 0 ..< t + 1:
          var score = 0.0f
          for d in 0 ..< headDim:
            score += cpuQ[h * headDim + d, t] * cpuK[hkv * headDim + d, i]
          scores[i, 0] = score * scale

        softmaxCols(scores)

        for i in 0 ..< t + 1:
          for d in 0 ..< headDim:
            attnOut[h * headDim + d, t] += scores[i, 0] * cpuV[hkv * headDim + d, i]

    let attnOutG = GGTensor(at: attnOut.toDevice())
    x.at = x.at + amMatmul(wo, attnOutG).at

    # FFN
    let ffnNorm = m.getTensor(prefix & "ffn_norm.weight")
    let w1 = m.getTensor(prefix & "ffn_gate.weight")
    let w2 = m.getTensor(prefix & "ffn_down.weight")
    let w3 = m.getTensor(prefix & "ffn_up.weight")

    let xNorm2 = rmsnormCols(x, ffnNorm, hp.rmsEps, hp.normType == "gemma")
    var g = amMatmul(w1, xNorm2)
    if hp.actType == "gelu":
      g = gelu(g)
    else:
      g = silu(g)
    let u = amMatmul(w3, xNorm2)
    let ffnOut = amMatmul(w2, GGTensor(at: (g.at * u.at)))
    x.at = x.at + ffnOut.at

  cache.curLen = tokens.len
  let normFinal = m.getTensor("output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, normFinal, hp.rmsEps, hp.normType == "gemma")
  let logits = amMatmul(outW, xNormFinal)

  # Return only last token's logits
  let lastLogits = logits.at[_, tokens.len - 1].reshape(hp.nVocab, 1)
  GGTensor(at: lastLogits)

proc forwardNext*(m: var Model, cache: var KvCache, token: int32): GGTensor =
  let hp = m.hparams
  var x = embeddingLookup(m.getTensor("token_embd.weight"), @[int(token)])
  let pos = cache.curLen

  let headDim = hp.headDim
  let ropeDim = hp.ropeDim

  for layer in 0 ..< hp.nLayer:
    let prefix = "blk." & $layer & "."
    let norm = m.getTensor(prefix & "attn_norm.weight")
    let wq = m.getTensor(prefix & "attn_q.weight")
    let wk = m.getTensor(prefix & "attn_k.weight")
    let wv = m.getTensor(prefix & "attn_v.weight")
    let wo = m.getTensor(prefix & "attn_output.weight")

    let xNorm = rmsnormCols(x, norm, hp.rmsEps, hp.normType == "gemma")
    var q = amMatmul(wq, xNorm)
    var k = amMatmul(wk, xNorm)
    var v = amMatmul(wv, xNorm)

    applyRopeInternal(q, hp.nHead, headDim, ropeDim, hp.ropeFreqBase, pos, hp.arch)
    applyRopeInternal(k, hp.nHeadKv, headDim, ropeDim, hp.ropeFreqBase, pos, hp.arch)

    cache.k[layer].at[_, pos .. pos] = k.at
    cache.v[layer].at[_, pos .. pos] = v.at

    let cpuQ = q.at.toCpu()
    let cpuKCache = cache.k[layer].at[_, 0 .. pos].toCpu()
    let cpuVCache = cache.v[layer].at[_, 0 .. pos].toCpu()

    var attnOut = newTensor[float32](hp.nEmb, 1)
    let nHead = hp.nHead
    let nHeadKv = hp.nHeadKv
    let nGroup = nHead div nHeadKv
    let scale = 1.0f / sqrt(float32(headDim))

    for h in 0 ..< nHead:
      let hkv = h div nGroup
      var scores = newTensor[float32](pos + 1, 1)
      for i in 0 ..< pos + 1:
        var score = 0.0f
        for d in 0 ..< headDim:
          score += cpuQ[h * headDim + d, 0] * cpuKCache[hkv * headDim + d, i]
        scores[i, 0] = score * scale

      softmaxCols(scores)

      for i in 0 ..< pos + 1:
        for d in 0 ..< headDim:
          attnOut[h * headDim + d, 0] += scores[i, 0] * cpuVCache[hkv * headDim + d, i]

    let attnOutG = GGTensor(at: attnOut.toDevice())
    x.at = x.at + amMatmul(wo, attnOutG).at

    let ffnNorm = m.getTensor(prefix & "ffn_norm.weight")
    let w1 = m.getTensor(prefix & "ffn_gate.weight")
    let w2 = m.getTensor(prefix & "ffn_down.weight")
    let w3 = m.getTensor(prefix & "ffn_up.weight")

    let xNorm2 = rmsnormCols(x, ffnNorm, hp.rmsEps, hp.normType == "gemma")
    var g = amMatmul(w1, xNorm2)
    if hp.actType == "gelu":
      g = gelu(g)
    else:
      g = silu(g)
    let u = amMatmul(w3, xNorm2)
    let ffnOut = amMatmul(w2, GGTensor(at: (g.at * u.at)))
    x.at = x.at + ffnOut.at

  cache.curLen += 1
  let normFinal = m.getTensor("output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, normFinal, hp.rmsEps, hp.normType == "gemma")
  result = amMatmul(outW, xNormFinal)
