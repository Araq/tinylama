#include <cstdint>
#include <cstring>

#if (defined(TINYLAMA_ENABLE_SYCL) || defined(SYCL_LANGUAGE_VERSION)) && (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))
  #if __has_include(<sycl/sycl.hpp>)
    #include <sycl/sycl.hpp>
    namespace tl_sycl = sycl;
  #else
    #include <CL/sycl.hpp>
    namespace tl_sycl = cl::sycl;
  #endif
  #define TINYLAMA_HAS_SYCL 1
#else
  #define TINYLAMA_HAS_SYCL 0
#endif

#if defined(TINYLAMA_REQUIRE_SYCL) && !TINYLAMA_HAS_SYCL
#error "useSycl is enabled but SYCL headers/toolchain are not available. Build with a SYCL compiler (e.g. icpx -fsycl)."
#endif

extern "C" int tinylama_sycl_backend_available() {
#if TINYLAMA_HAS_SYCL
  return 1;
#else
  return 0;
#endif
}

extern "C" int tinylama_sycl_linear_f32(
  const float * x,
  const float * w,
  float * out,
  int in_dim,
  int seq_len,
  int out_rows
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue q{tl_sycl::default_selector_v};

    tl_sycl::buffer<float, 2> xbuf(x, tl_sycl::range<2>((size_t) in_dim, (size_t) seq_len));
    tl_sycl::buffer<float, 2> wbuf(w, tl_sycl::range<2>((size_t) out_rows, (size_t) in_dim));
    tl_sycl::buffer<float, 2> obuf(out, tl_sycl::range<2>((size_t) out_rows, (size_t) seq_len));

    q.submit([&](tl_sycl::handler & h) {
      auto xa = xbuf.get_access<tl_sycl::access::mode::read>(h);
      auto wa = wbuf.get_access<tl_sycl::access::mode::read>(h);
      auto oa = obuf.get_access<tl_sycl::access::mode::write>(h);

      h.parallel_for(tl_sycl::range<2>((size_t) out_rows, (size_t) seq_len),
                     [=](tl_sycl::id<2> idx) {
        const int o = (int) idx[0];
        const int s = (int) idx[1];
        float acc = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
          acc += wa[(size_t) o][(size_t) k] * xa[(size_t) k][(size_t) s];
        }
        oa[(size_t) o][(size_t) s] = acc;
      });
    });

    q.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) x;
  (void) w;
  (void) out;
  (void) in_dim;
  (void) seq_len;
  (void) out_rows;
  return 0;
#endif
}

extern "C" int tinylama_sycl_rmsnorm_cols_f32(
  const float * x,
  const float * weight,
  float * out,
  int dim,
  int seq_len,
  float eps
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue q{tl_sycl::default_selector_v};

    tl_sycl::buffer<float, 2> xbuf(x, tl_sycl::range<2>((size_t) dim, (size_t) seq_len));
    tl_sycl::buffer<float, 1> wbuf(weight, tl_sycl::range<1>((size_t) dim));
    tl_sycl::buffer<float, 2> obuf(out, tl_sycl::range<2>((size_t) dim, (size_t) seq_len));

    q.submit([&](tl_sycl::handler & h) {
      auto xa = xbuf.get_access<tl_sycl::access::mode::read>(h);
      auto wa = wbuf.get_access<tl_sycl::access::mode::read>(h);
      auto oa = obuf.get_access<tl_sycl::access::mode::write>(h);

      h.parallel_for(tl_sycl::range<1>((size_t) seq_len), [=](tl_sycl::id<1> id) {
        const int s = (int) id[0];
        float ss = 0.0f;
        for (int d = 0; d < dim; ++d) {
          const float v = xa[(size_t) d][(size_t) s];
          ss += v * v;
        }
        const float inv = 1.0f / tl_sycl::sqrt(ss / (float) dim + eps);
        for (int d = 0; d < dim; ++d) {
          oa[(size_t) d][(size_t) s] = xa[(size_t) d][(size_t) s] * inv * wa[(size_t) d];
        }
      });
    });

    q.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) x;
  (void) weight;
  (void) out;
  (void) dim;
  (void) seq_len;
  (void) eps;
  return 0;
#endif
}

extern "C" int tinylama_sycl_store_kv_cols_f32(
  float * cache,
  const float * src,
  int rows,
  int src_cols,
  int cache_cols,
  int start_pos
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue q{tl_sycl::default_selector_v};
    const size_t total = (size_t) rows * (size_t) src_cols;

    tl_sycl::buffer<float, 1> cbuf(cache, tl_sycl::range<1>((size_t) rows * (size_t) cache_cols));
    tl_sycl::buffer<float, 1> sbuf(src, tl_sycl::range<1>(total));

    q.submit([&](tl_sycl::handler & h) {
      auto ca = cbuf.get_access<tl_sycl::access::mode::read_write>(h);
      auto sa = sbuf.get_access<tl_sycl::access::mode::read>(h);

      h.parallel_for(tl_sycl::range<1>(total), [=](tl_sycl::id<1> id) {
        const size_t idx = id[0];
        const int r = (int) (idx / (size_t) src_cols);
        const int c = (int) (idx % (size_t) src_cols);
        const size_t dst = (size_t) r * (size_t) cache_cols + (size_t) (start_pos + c);
        ca[dst] = sa[idx];
      });
    });

    q.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) cache;
  (void) src;
  (void) rows;
  (void) src_cols;
  (void) cache_cols;
  (void) start_pos;
  return 0;
#endif
}

extern "C" int tinylama_sycl_attention_decode_f32(
  const float * q,
  const float * kcache,
  const float * vcache,
  float * out,
  int n_head,
  int n_head_kv,
  int head_dim,
  int cur_len,
  int cache_cols
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue qu{tl_sycl::default_selector_v};
    const int out_dim = n_head * head_dim;
    const int group = n_head / n_head_kv;

    tl_sycl::buffer<float, 1> qbuf(q, tl_sycl::range<1>((size_t) out_dim));
    tl_sycl::buffer<float, 1> kbuf(kcache, tl_sycl::range<1>((size_t) n_head_kv * (size_t) head_dim * (size_t) cache_cols));
    tl_sycl::buffer<float, 1> vbuf(vcache, tl_sycl::range<1>((size_t) n_head_kv * (size_t) head_dim * (size_t) cache_cols));
    tl_sycl::buffer<float, 1> obuf(out, tl_sycl::range<1>((size_t) out_dim));

    qu.submit([&](tl_sycl::handler & h) {
      auto qa = qbuf.get_access<tl_sycl::access::mode::read>(h);
      auto ka = kbuf.get_access<tl_sycl::access::mode::read>(h);
      auto va = vbuf.get_access<tl_sycl::access::mode::read>(h);
      auto oa = obuf.get_access<tl_sycl::access::mode::write>(h);

      h.parallel_for(tl_sycl::range<1>((size_t) out_dim), [=](tl_sycl::id<1> id) {
        const int od = (int) id[0];
        const int hq = od / head_dim;
        const int d = od % head_dim;
        const int kvh = hq / group;
        const float inv = 1.0f / tl_sycl::sqrt((float) head_dim);

        // Pass 1: find max score for softmax stability.
        float maxScore = -3.4028235e38f;
        for (int j = 0; j < cur_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = hq * head_dim + kk;
            const int kIdx = (kvh * head_dim + kk) * cache_cols + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          const float s = dot * inv;
          if (s > maxScore) {
            maxScore = s;
          }
        }

        // Pass 2: softmax denominator.
        float sum = 0.0f;
        for (int j = 0; j < cur_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = hq * head_dim + kk;
            const int kIdx = (kvh * head_dim + kk) * cache_cols + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          sum += tl_sycl::exp(dot * inv - maxScore);
        }
        const float invSum = 1.0f / sum;

        // Pass 3: weighted sum on V for this output element.
        float acc = 0.0f;
        for (int j = 0; j < cur_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = hq * head_dim + kk;
            const int kIdx = (kvh * head_dim + kk) * cache_cols + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          const float p = tl_sycl::exp(dot * inv - maxScore) * invSum;
          const int vIdx = (kvh * head_dim + d) * cache_cols + j;
          acc += p * va[(size_t) vIdx];
        }
        oa[(size_t) od] = acc;
      });
    });

    qu.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) q;
  (void) kcache;
  (void) vcache;
  (void) out;
  (void) n_head;
  (void) n_head_kv;
  (void) head_dim;
  (void) cur_len;
  (void) cache_cols;
  return 0;
#endif
}

extern "C" int tinylama_sycl_rope_at_pos_f32(
  float * x,
  int n_head,
  int head_dim,
  int rope_dim,
  float base,
  int pos,
  int seq_len
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue qu{tl_sycl::default_selector_v};
    const int total = n_head * (rope_dim / 2) * seq_len;

    tl_sycl::buffer<float, 1> xbuf(x, tl_sycl::range<1>((size_t) n_head * (size_t) head_dim * (size_t) seq_len));

    qu.submit([&](tl_sycl::handler & h) {
      auto xa = xbuf.get_access<tl_sycl::access::mode::read_write>(h);

      h.parallel_for(tl_sycl::range<1>((size_t) total), [=](tl_sycl::id<1> id) {
        const int idx = (int) id[0];
        const int half = rope_dim / 2;
        const int perHead = half * seq_len;
        const int hq = idx / perHead;
        const int rem = idx % perHead;
        const int i = rem / seq_len;
        const int p = rem % seq_len;

        const float theta = tl_sycl::pow(1.0f / base, (2.0f * (float) i) / (float) rope_dim);
        const float angle = (seq_len == 1 ? (float) pos : (float) p) * theta;
        const float c = tl_sycl::cos(angle);
        const float s = tl_sycl::sin(angle);

        const int idx0 = ((hq * head_dim + 2 * i) * seq_len) + p;
        const int idx1 = ((hq * head_dim + 2 * i + 1) * seq_len) + p;
        const float v0 = xa[(size_t) idx0];
        const float v1 = xa[(size_t) idx1];
        xa[(size_t) idx0] = v0 * c - v1 * s;
        xa[(size_t) idx1] = v0 * s + v1 * c;
      });
    });

    qu.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) x;
  (void) n_head;
  (void) head_dim;
  (void) rope_dim;
  (void) base;
  (void) pos;
  (void) seq_len;
  return 0;
#endif
}

extern "C" int tinylama_sycl_attention_prefill_f32(
  const float * q,
  const float * k,
  const float * v,
  float * out,
  int n_head,
  int n_head_kv,
  int head_dim,
  int seq_len
) {
#if TINYLAMA_HAS_SYCL
  try {
    tl_sycl::queue qu{tl_sycl::default_selector_v};
    const int out_dim = n_head * head_dim * seq_len;
    const int group = n_head / n_head_kv;

    tl_sycl::buffer<float, 1> qbuf(q, tl_sycl::range<1>((size_t) n_head * (size_t) head_dim * (size_t) seq_len));
    tl_sycl::buffer<float, 1> kbuf(k, tl_sycl::range<1>((size_t) n_head_kv * (size_t) head_dim * (size_t) seq_len));
    tl_sycl::buffer<float, 1> vbuf(v, tl_sycl::range<1>((size_t) n_head_kv * (size_t) head_dim * (size_t) seq_len));
    tl_sycl::buffer<float, 1> obuf(out, tl_sycl::range<1>((size_t) out_dim));

    qu.submit([&](tl_sycl::handler & h) {
      auto qa = qbuf.get_access<tl_sycl::access::mode::read>(h);
      auto ka = kbuf.get_access<tl_sycl::access::mode::read>(h);
      auto va = vbuf.get_access<tl_sycl::access::mode::read>(h);
      auto oa = obuf.get_access<tl_sycl::access::mode::write>(h);

      h.parallel_for(tl_sycl::range<1>((size_t) out_dim), [=](tl_sycl::id<1> id) {
        const int od = (int) id[0];
        const int perHead = head_dim * seq_len;
        const int hq = od / perHead;
        const int rem = od % perHead;
        const int d = rem / seq_len;
        const int qi = rem % seq_len;
        const int kvh = hq / group;
        const int causal_len = qi + 1;
        const float inv = 1.0f / tl_sycl::sqrt((float) head_dim);

        float maxScore = -3.4028235e38f;
        for (int j = 0; j < causal_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = (hq * head_dim + kk) * seq_len + qi;
            const int kIdx = (kvh * head_dim + kk) * seq_len + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          const float s = dot * inv;
          if (s > maxScore) {
            maxScore = s;
          }
        }

        float sum = 0.0f;
        for (int j = 0; j < causal_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = (hq * head_dim + kk) * seq_len + qi;
            const int kIdx = (kvh * head_dim + kk) * seq_len + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          sum += tl_sycl::exp(dot * inv - maxScore);
        }
        const float invSum = 1.0f / sum;

        float acc = 0.0f;
        for (int j = 0; j < causal_len; ++j) {
          float dot = 0.0f;
          for (int kk = 0; kk < head_dim; ++kk) {
            const int qIdx = (hq * head_dim + kk) * seq_len + qi;
            const int kIdx = (kvh * head_dim + kk) * seq_len + j;
            dot += qa[(size_t) qIdx] * ka[(size_t) kIdx];
          }
          const float p = tl_sycl::exp(dot * inv - maxScore) * invSum;
          const int vIdx = (kvh * head_dim + d) * seq_len + j;
          acc += p * va[(size_t) vIdx];
        }
        oa[(size_t) od] = acc;
      });
    });

    qu.wait_and_throw();
    return 1;
  } catch (...) {
    return 0;
  }
#else
  (void) q;
  (void) k;
  (void) v;
  (void) out;
  (void) n_head;
  (void) n_head_kv;
  (void) head_dim;
  (void) seq_len;
  return 0;
#endif
}

