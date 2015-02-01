/*===---- avx512fintrin.h - AVX2 intrinsics -----------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error "Never use <avx512erintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512ERINTRIN_H
#define __AVX512ERINTRIN_H


// rsqrt28
#define _mm512_rsqrt28_round_pd(A, R) __extension ({ \
  (__m512d)__builtin_ia32_rsqrt28pd_mask((__v8df)(__m512d)(A), \
                                         (__v8df)_mm512_setzero_pd(), \
                                         (__mmask8)-1, (R)); })

#define _mm512_rsqrt28_round_ps(A, R) __extension__ ({ \
  (__m512)__builtin_ia32_rsqrt28ps_mask((__v16sf)(__m512)(A), \
                                        (__v16sf)_mm512_setzero_ps(), \
                                        (__mmask16)-1, (R)); })

#define _mm_rsqrt28_round_ss(A, B, R) __extension__ ({ \
  (__m128)__builtin_ia32_rsqrt28ss_mask((__v4sf)(__m128)(A), \
                                        (__v4sf)(__m128(B), \
                                        (__v4sf)_mm_setzero_ps(), \
                                        (__mmask8)-1, (R)); })

#define _mm_rsqrt28_round_sd (A, B, R) __extension__ ({ \
  (__m128d)__builtin_ia32_rsqrt28sd_mask((__v2df)(__m128d)(A), \
                                         (__v2df)(__m128d)(B), \
                                         (__v2df)_mm_setzero_pd(), \
                                         (__mmask8)-1, (R)); })


// rcp28
#define _mm512_rcp28_round_pd(A, R) __extension__ ({ \
  (__m512d)__builtin_ia32_rcp28pd_mask((__v8df)(__m512d)(A), \
                                       (__v8df)_mm512_setzero_pd(), \
                                       (__mmask8)-1, (R)); })

#define _mm512_rcp28_round_ps(A, R) __extension__ ({ \
  (__m512)__builtin_ia32_rcp28ps_mask((__v16sf)(__m512)(A), \
                                      (__v16sf)_mm512_setzero_ps(), \
                                      (__mmask16)-1, (R)); })

#define _mm_rcp28_round_ss(A, B, R) __extension__ ({ \
  (__m128) __builtin_ia32_rcp28ss_mask ((__v4sf)(__m128)(A), \
                                        (__v4sf)(__m128)(B), \
                                        (__v4sf)_mm_setzero_ps(), \
                                        (__mmask8)-1, (R)); })

#define _mm_rcp28_round_sd(A, B, R) __extension__ ({ \
  (__m128d)__builtin_ia32_rcp28sd_mask((__v2df)(__m128d)(A), \
                                       (__v2df)(__m128d)(B), \
                                       (__v2df)_mm_setzero_pd(), \
                                       (__mmask8)-1, (R)); })

#endif // __AVX512ERINTRIN_H
