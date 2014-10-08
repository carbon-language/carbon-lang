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
_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqq256_mask((__v4di)__a, (__v4di)__b,
                                                  __u);
}

#endif /* __AVX512VLINTRIN_H */
