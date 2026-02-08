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

    # Attention (Prefill - Optimized)
    let nHead = hp.nHead
    let nHeadKv = hp.nHeadKv
    let nGroup = nHead div nHeadKv
    let scale = 1.0f / sqrt(float32(headDim))
    let N = tokens.len

    let q_ext = q.at.toCpu().reshape(nHead, headDim, N)
    let k_ext = k.at.toCpu().reshape(nHeadKv, headDim, N)
    let v_ext = v.at.toCpu().reshape(nHeadKv, headDim, N)

    var attnOut = newTensor[float32](nHead, headDim, N)

    for h in 0 ..< nHead:
      let hkv = h div nGroup
      let qh = q_ext[h, _, _].reshape(headDim, N)
      let kh = k_ext[hkv, _, _].reshape(headDim, N)
      let vh = v_ext[hkv, _, _].reshape(headDim, N)

      var scores = (qh.transpose() * kh) *. scale
      for i in 0 ..< N:
        for j in i + 1 ..< N:
          scores[i, j] = -1e30f

      var scores_sm = newTensor[float32](N, N)
      for i in 0 ..< N:
        let row = scores[i, _].reshape(N)
        scores_sm[i, _] = row.softmax().reshape(1, N)
      let oh = vh * scores_sm.transpose()
      attnOut[h, _, _] = oh.reshape(1, headDim, N)

    let attnOutG = GGTensor(at: attnOut.reshape(nHead * headDim, N).toDevice())
    x.at = x.at +. amMatmul(wo, attnOutG).at

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
    let ffnOut = amMatmul(w2, GGTensor(at: (g.at *. u.at)))
    x.at = x.at +. ffnOut.at

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


    # Attention (Next - Optimized)
    let nHead = hp.nHead
    let nHeadKv = hp.nHeadKv
    let nGroup = nHead div nHeadKv
    let scale = 1.0f / sqrt(float32(headDim))

    let q_ext = q.at.toCpu().reshape(nHead, headDim, 1)
    let k_ext = cache.k[layer].at[_, 0 .. pos].toCpu().reshape(nHeadKv, headDim, pos + 1)
    let v_ext = cache.v[layer].at[_, 0 .. pos].toCpu().reshape(nHeadKv, headDim, pos + 1)

    var attnOut = newTensor[float32](nHead, headDim, 1)

    for h in 0 ..< nHead:
      let hkv = h div nGroup
      let qh = q_ext[h, _, _].reshape(headDim, 1)
      let kh = k_ext[hkv, _, _].reshape(headDim, pos + 1)
      let vh = v_ext[hkv, _, _].reshape(headDim, pos + 1)

      let scores = (qh.transpose() * kh) *. scale
      let scores_sm = scores.softmax()
      let oh = vh * scores_sm.transpose()
      attnOut[h, _, _] = oh.reshape(1, headDim, 1)

    let attnOutG = GGTensor(at: attnOut.toDevice())
    x.at = x.at +. amMatmul(wo, attnOutG).at

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
    let ffnOut = amMatmul(w2, GGTensor(at: (g.at *. u.at)))
    x.at = x.at +. ffnOut.at

  cache.curLen += 1
  let normFinal = m.getTensor("output_norm.weight")
  let outW = m.getTensor("output.weight")
  let xNormFinal = rmsnormCols(x, normFinal, hp.rmsEps, hp.normType == "gemma")
  result = amMatmul(outW, xNormFinal)
