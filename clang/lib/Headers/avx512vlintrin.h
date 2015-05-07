/*===---- avx512vlintrin.h - AVX512VL intrinsics ---------------------------===
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
#error "Never use <avx512vlintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLINTRIN_H
#define __AVX512VLINTRIN_H

/* Integer compare */

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd128_mask((__v4si)__a, (__v4si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd128_mask((__v4si)__a, (__v4si)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd256_mask((__v8si)__a, (__v8si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqd256_mask((__v8si)__a, (__v8si)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq128_mask((__v2di)__a, (__v2di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq128_mask((__v2di)__a, (__v2di)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 0,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 0,
                                                __u);
}


static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 5,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 5,
                                                __u);
}




static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd128_mask((__v4si)__a, (__v4si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd128_mask((__v4si)__a, (__v4si)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd256_mask((__v8si)__a, (__v8si)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtd256_mask((__v8si)__a, (__v8si)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq128_mask((__v2di)__a, (__v2di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq128_mask((__v2di)__a, (__v2di)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq256_mask((__v4di)__a, (__v4di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtq256_mask((__v4di)__a, (__v4di)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 6,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 2,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 1,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)__a, (__v4si)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)__a, (__v8si)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)__a, (__v2di)__b, 4,
                                                __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)__a, (__v4di)__b, 4,
                                                __u);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_add_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_paddd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_add_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_paddd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_add_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_paddq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_add_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_paddq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_sub_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_psubd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_sub_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_psubd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_sub_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_psubq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_sub_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_psubq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_add_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_paddd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_add_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_paddd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_add_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_paddq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_add_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_paddq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_sub_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psubd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_sub_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psubd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_sub_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_psubq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_sub_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_psubq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_mul_epi32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuldq256_mask ((__v8si) __X,
              (__v8si) __Y,
              (__v4di) __W, __M);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_mul_epi32 (__mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuldq256_mask ((__v8si) __X,
              (__v8si) __Y,
              (__v4di)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_mul_epi32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuldq128_mask ((__v4si) __X,
              (__v4si) __Y,
              (__v2di) __W, __M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_mul_epi32 (__mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuldq128_mask ((__v4si) __X,
              (__v4si) __Y,
              (__v2di)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_mul_epu32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuludq256_mask ((__v8si) __X,
               (__v8si) __Y,
               (__v4di) __W, __M);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_mul_epu32 (__mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i) __builtin_ia32_pmuludq256_mask ((__v8si) __X,
               (__v8si) __Y,
               (__v4di)
               _mm256_setzero_si256 (),
               __M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_mul_epu32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuludq128_mask ((__v4si) __X,
               (__v4si) __Y,
               (__v2di) __W, __M);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_mul_epu32 (__mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i) __builtin_ia32_pmuludq128_mask ((__v4si) __X,
               (__v4si) __Y,
               (__v2di)
               _mm_setzero_si128 (),
               __M);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_mullo_epi32 (__mmask8 __M, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pmulld256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              __M);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_mullo_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
       __m256i __B)
{
  return (__m256i) __builtin_ia32_pmulld256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W, __M);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_mullo_epi32 (__mmask8 __M, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pmulld128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              __M);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_mullo_epi32 (__m128i __W, __mmask16 __M, __m128i __A,
          __m128i __B)
{
  return (__m128i) __builtin_ia32_pmulld128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W, __M);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_and_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pandd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandd256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandd128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si) __W,
              (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnd256_mask ((__v8si) __A,
              (__v8si) __B,
              (__v8si)
              _mm256_setzero_si256 (),
              (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnd128_mask ((__v4si) __A,
              (__v4si) __B,
              (__v4si)
              _mm_setzero_si128 (),
              (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B)
{
  return (__m256i) __builtin_ia32_pord256_mask ((__v8si) __A,
            (__v8si) __B,
            (__v8si) __W,
            (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pord256_mask ((__v8si) __A,
            (__v8si) __B,
            (__v8si)
            _mm256_setzero_si256 (),
            (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pord128_mask ((__v4si) __A,
            (__v4si) __B,
            (__v4si) __W,
            (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pord128_mask ((__v4si) __A,
            (__v4si) __B,
            (__v4si)
            _mm_setzero_si128 (),
            (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pxord256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pxord256_mask ((__v8si) __A,
             (__v8si) __B,
             (__v8si)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pxord128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pxord128_mask ((__v4si) __A,
             (__v4si) __B,
             (__v4si)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pandq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W, __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_pd (),
             __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pandq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W, __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_pd (),
             __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di) __W, __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pandnq256_mask ((__v4di) __A,
              (__v4di) __B,
              (__v4di)
              _mm256_setzero_pd (),
              __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di) __W, __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pandnq128_mask ((__v2di) __A,
              (__v2di) __B,
              (__v2di)
              _mm_setzero_pd (),
              __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B)
{
  return (__m256i) __builtin_ia32_porq256_mask ((__v4di) __A,
            (__v4di) __B,
            (__v4di) __W,
            (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_porq256_mask ((__v4di) __A,
            (__v4di) __B,
            (__v4di)
            _mm256_setzero_si256 (),
            (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_porq128_mask ((__v2di) __A,
            (__v2di) __B,
            (__v2di) __W,
            (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_porq128_mask ((__v2di) __A,
            (__v2di) __B,
            (__v2di)
            _mm_setzero_si128 (),
            (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B)
{
  return (__m256i) __builtin_ia32_pxorq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di) __W,
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_pxorq256_mask ((__v4di) __A,
             (__v4di) __B,
             (__v4di)
             _mm256_setzero_si256 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B)
{
  return (__m128i) __builtin_ia32_pxorq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B)
{
  return (__m128i) __builtin_ia32_pxorq128_mask ((__v2di) __A,
             (__v2di) __B,
             (__v2di)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

#define _mm_cmp_epi32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)(__m128i)(a), \
                                        (__v4si)(__m128i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm_mask_cmp_epi32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd128_mask((__v4si)(__m128i)(a), \
                                        (__v4si)(__m128i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm_cmp_epu32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)(__m128i)(a), \
                                         (__v4si)(__m128i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm_mask_cmp_epu32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd128_mask((__v4si)(__m128i)(a), \
                                         (__v4si)(__m128i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_epi32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)(__m256i)(a), \
                                        (__v8si)(__m256i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epi32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpd256_mask((__v8si)(__m256i)(a), \
                                        (__v8si)(__m256i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm256_cmp_epu32_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)(__m256i)(a), \
                                         (__v8si)(__m256i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epu32_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpd256_mask((__v8si)(__m256i)(a), \
                                         (__v8si)(__m256i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm_cmp_epi64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)(__m128i)(a), \
                                        (__v2di)(__m128i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm_mask_cmp_epi64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq128_mask((__v2di)(__m128i)(a), \
                                        (__v2di)(__m128i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm_cmp_epu64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)(__m128i)(a), \
                                         (__v2di)(__m128i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm_mask_cmp_epu64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq128_mask((__v2di)(__m128i)(a), \
                                         (__v2di)(__m128i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_epi64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)(__m256i)(a), \
                                        (__v4di)(__m256i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epi64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpq256_mask((__v4di)(__m256i)(a), \
                                        (__v4di)(__m256i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm256_cmp_epu64_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)(__m256i)(a), \
                                         (__v4di)(__m256i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_epu64_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpq256_mask((__v4di)(__m256i)(a), \
                                         (__v4di)(__m256i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_ps_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps256_mask((__v8sf)(__m256)(a), \
                                         (__v8sf)(__m256)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_ps_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps256_mask((__v8sf)(__m256)(a), \
                                         (__v8sf)(__m256)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_pd_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd256_mask((__v4df)(__m256)(a), \
                                         (__v4df)(__m256)(b), \
                                         (p), (__mmask8)-1); })

#define _mm256_mask_cmp_pd_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd256_mask((__v4df)(__m256)(a), \
                                         (__v4df)(__m256)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm128_cmp_ps_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps128_mask((__v4sf)(__m128)(a), \
                                         (__v4sf)(__m128)(b), \
                                         (p), (__mmask8)-1); })

#define _mm128_mask_cmp_ps_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpps128_mask((__v4sf)(__m128)(a), \
                                         (__v4sf)(__m128)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm128_cmp_pd_mask(a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd128_mask((__v2df)(__m128)(a), \
                                         (__v2df)(__m128)(b), \
                                         (p), (__mmask8)-1); })

#define _mm128_mask_cmp_pd_mask(m, a, b, p)  __extension__ ({ \
  (__mmask8)__builtin_ia32_cmppd128_mask((__v2df)(__m128)(a), \
                                         (__v2df)(__m128)(b), \
                                         (p), (__mmask8)(m)); })
#endif /* __AVX512VLINTRIN_H */
