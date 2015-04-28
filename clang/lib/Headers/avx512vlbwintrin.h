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

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 0,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 0,
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

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 0,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 0,
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

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 0,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpeq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 0,
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

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 0,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpeq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 0,
                                                 __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 5,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 5,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 5,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 5,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 5,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 5,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 5,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 5,
                                                 __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 5,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 5,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 5,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpge_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 5,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 5,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 5,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpge_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 5,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpge_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 5,
                                                 __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_pcmpgtb128_mask((__v16qi)__a, (__v16qi)__b,
                                                   (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_pcmpgtb128_mask((__v16qi)__a, (__v16qi)__b,
                                                   __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 6,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 6,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_pcmpgtb256_mask((__v32qi)__a, (__v32qi)__b,
                                                   (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_pcmpgtb256_mask((__v32qi)__a, (__v32qi)__b,
                                                   __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 6,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 6,
                                                 __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtw128_mask((__v8hi)__a, (__v8hi)__b,
                                                  (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_pcmpgtw128_mask((__v8hi)__a, (__v8hi)__b,
                                                  __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 6,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpgt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 6,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_pcmpgtw256_mask((__v16hi)__a, (__v16hi)__b,
                                                   (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_pcmpgtw256_mask((__v16hi)__a, (__v16hi)__b,
                                                   __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 6,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpgt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 6,
                                                 __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 2,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 2,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 2,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 2,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 2,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 2,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 2,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 2,
                                                 __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 2,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 2,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 2,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmple_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 2,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 2,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 2,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmple_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 2,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmple_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 2,
                                                 __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 1,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 1,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 1,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 1,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 1,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 1,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 1,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 1,
                                                 __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 1,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 1,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 1,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmplt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 1,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 1,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 1,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmplt_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 1,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmplt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 1,
                                                 __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epi8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 4,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)__a, (__v16qi)__b, 4,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epu8_mask(__m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 4,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  return (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)__a, (__v16qi)__b, 4,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epi8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 4,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)__a, (__v32qi)__b, 4,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epu8_mask(__m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 4,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  return (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)__a, (__v32qi)__b, 4,
                                                 __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epi16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 4,
                                               (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)__a, (__v8hi)__b, 4,
                                               __u);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_epu16_mask(__m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 4,
                                                (__mmask8)-1);
}

static __inline__ __mmask8 __attribute__((__always_inline__, __nodebug__))
_mm_mask_cmpneq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  return (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)__a, (__v8hi)__b, 4,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epi16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 4,
                                                (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)__a, (__v16hi)__b, 4,
                                                __u);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_cmpneq_epu16_mask(__m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 4,
                                                 (__mmask16)-1);
}

static __inline__ __mmask16 __attribute__((__always_inline__, __nodebug__))
_mm256_mask_cmpneq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  return (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)__a, (__v16hi)__b, 4,
                                                 __u);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_add_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B){
  return (__m256i) __builtin_ia32_paddb256_mask ((__v32qi) __A,
             (__v32qi) __B,
             (__v32qi) __W,
             (__mmask32) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_add_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_paddb256_mask ((__v32qi) __A,
             (__v32qi) __B,
             (__v32qi)
             _mm256_setzero_si256 (),
             (__mmask32) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_add_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_paddw256_mask ((__v16hi) __A,
             (__v16hi) __B,
             (__v16hi) __W,
             (__mmask16) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_add_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_paddw256_mask ((__v16hi) __A,
             (__v16hi) __B,
             (__v16hi)
             _mm256_setzero_si256 (),
             (__mmask16) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_sub_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_psubb256_mask ((__v32qi) __A,
             (__v32qi) __B,
             (__v32qi) __W,
             (__mmask32) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_sub_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_psubb256_mask ((__v32qi) __A,
             (__v32qi) __B,
             (__v32qi)
             _mm256_setzero_si256 (),
             (__mmask32) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mask_sub_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_psubw256_mask ((__v16hi) __A,
             (__v16hi) __B,
             (__v16hi) __W,
             (__mmask16) __U);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maskz_sub_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_psubw256_mask ((__v16hi) __A,
             (__v16hi) __B,
             (__v16hi)
             _mm256_setzero_si256 (),
             (__mmask16) __U);
}
static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_add_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_paddb128_mask ((__v16qi) __A,
             (__v16qi) __B,
             (__v16qi) __W,
             (__mmask16) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_add_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_paddb128_mask ((__v16qi) __A,
             (__v16qi) __B,
             (__v16qi)
             _mm_setzero_si128 (),
             (__mmask16) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_add_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_paddw128_mask ((__v8hi) __A,
             (__v8hi) __B,
             (__v8hi) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_add_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_paddw128_mask ((__v8hi) __A,
             (__v8hi) __B,
             (__v8hi)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_sub_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_psubb128_mask ((__v16qi) __A,
             (__v16qi) __B,
             (__v16qi) __W,
             (__mmask16) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_sub_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_psubb128_mask ((__v16qi) __A,
             (__v16qi) __B,
             (__v16qi)
             _mm_setzero_si128 (),
             (__mmask16) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mask_sub_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_psubw128_mask ((__v8hi) __A,
             (__v8hi) __B,
             (__v8hi) __W,
             (__mmask8) __U);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maskz_sub_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_psubw128_mask ((__v8hi) __A,
             (__v8hi) __B,
             (__v8hi)
             _mm_setzero_si128 (),
             (__mmask8) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_mask_mullo_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmullw256_mask ((__v16hi) __A,
              (__v16hi) __B,
              (__v16hi) __W,
              (__mmask16) __U);
}

static __inline__ __m256i __attribute__ ((__always_inline__, __nodebug__))
_mm256_maskz_mullo_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  return (__m256i) __builtin_ia32_pmullw256_mask ((__v16hi) __A,
              (__v16hi) __B,
              (__v16hi)
              _mm256_setzero_si256 (),
              (__mmask16) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_mask_mullo_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmullw128_mask ((__v8hi) __A,
              (__v8hi) __B,
              (__v8hi) __W,
              (__mmask8) __U);
}

static __inline__ __m128i __attribute__ ((__always_inline__, __nodebug__))
_mm_maskz_mullo_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  return (__m128i) __builtin_ia32_pmullw128_mask ((__v8hi) __A,
              (__v8hi) __B,
              (__v8hi)
              _mm_setzero_si128 (),
              (__mmask8) __U);
}
#define _mm_cmp_epi8_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)(__m128i)(a), \
                                         (__v16qi)(__m128i)(b), \
                                         (p), (__mmask16)-1); })

#define _mm_mask_cmp_epi8_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpb128_mask((__v16qi)(__m128i)(a), \
                                         (__v16qi)(__m128i)(b), \
                                         (p), (__mmask16)(m)); })

#define _mm_cmp_epu8_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)(__m128i)(a), \
                                          (__v16qi)(__m128i)(b), \
                                          (p), (__mmask16)-1); })

#define _mm_mask_cmp_epu8_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpb128_mask((__v16qi)(__m128i)(a), \
                                          (__v16qi)(__m128i)(b), \
                                          (p), (__mmask16)(m)); })

#define _mm256_cmp_epi8_mask(a, b, p) __extension__ ({ \
  (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)(__m256i)(a), \
                                         (__v32qi)(__m256i)(b), \
                                         (p), (__mmask32)-1); })

#define _mm256_mask_cmp_epi8_mask(m, a, b, p) __extension__ ({ \
  (__mmask32)__builtin_ia32_cmpb256_mask((__v32qi)(__m256i)(a), \
                                         (__v32qi)(__m256i)(b), \
                                         (p), (__mmask32)(m)); })

#define _mm256_cmp_epu8_mask(a, b, p) __extension__ ({ \
  (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)(__m256i)(a), \
                                          (__v32qi)(__m256i)(b), \
                                          (p), (__mmask32)-1); })

#define _mm256_mask_cmp_epu8_mask(m, a, b, p) __extension__ ({ \
  (__mmask32)__builtin_ia32_ucmpb256_mask((__v32qi)(__m256i)(a), \
                                          (__v32qi)(__m256i)(b), \
                                          (p), (__mmask32)(m)); })

#define _mm_cmp_epi16_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)(__m128i)(a), \
                                        (__v8hi)(__m128i)(b), \
                                        (p), (__mmask8)-1); })

#define _mm_mask_cmp_epi16_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_cmpw128_mask((__v8hi)(__m128i)(a), \
                                        (__v8hi)(__m128i)(b), \
                                        (p), (__mmask8)(m)); })

#define _mm_cmp_epu16_mask(a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)(__m128i)(a), \
                                         (__v8hi)(__m128i)(b), \
                                         (p), (__mmask8)-1); })

#define _mm_mask_cmp_epu16_mask(m, a, b, p) __extension__ ({ \
  (__mmask8)__builtin_ia32_ucmpw128_mask((__v8hi)(__m128i)(a), \
                                         (__v8hi)(__m128i)(b), \
                                         (p), (__mmask8)(m)); })

#define _mm256_cmp_epi16_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)(__m256i)(a), \
                                         (__v16hi)(__m256i)(b), \
                                         (p), (__mmask16)-1); })

#define _mm256_mask_cmp_epi16_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpw256_mask((__v16hi)(__m256i)(a), \
                                         (__v16hi)(__m256i)(b), \
                                         (p), (__mmask16)(m)); })

#define _mm256_cmp_epu16_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)(__m256i)(a), \
                                          (__v16hi)(__m256i)(b), \
                                          (p), (__mmask16)-1); })

#define _mm256_mask_cmp_epu16_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpw256_mask((__v16hi)(__m256i)(a), \
                                          (__v16hi)(__m256i)(b), \
                                          (p), (__mmask16)(m)); })

#endif /* __AVX512VLBWINTRIN_H */
