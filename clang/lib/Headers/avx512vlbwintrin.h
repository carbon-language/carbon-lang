/*===---- avx512vlbwintrin.h - AVX512VL and AVX512BW intrinsics ----------===
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
#error "Never use <avx512vlbwintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLBWINTRIN_H
#define __AVX512VLBWINTRIN_H

/* Integer compare */

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_pcmpeqb128_mask((__v16qi)__a, (__v16qi)__b,
                                                   (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_pcmpeqb128_mask((__v16qi)__a, (__v16qi)__b,
                                                   __u);
}


static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_pcmpeqb256_mask((__v32qi)__a, (__v32qi)__b,
                                                   (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_pcmpeqb256_mask((__v32qi)__a, (__v32qi)__b,
                                                   __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqw128_mask((__v8hi)__a, (__v8hi)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpeqw128_mask((__v8hi)__a, (__v8hi)__b,
                                                  __u);
}


static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_pcmpeqw256_mask((__v16hi)__a, (__v16hi)__b,
                                                   (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_pcmpeqw256_mask((__v16hi)__a, (__v16hi)__b,
                                                   __u);
}

#endif /* __AVX512VLBWINTRIN_H */
