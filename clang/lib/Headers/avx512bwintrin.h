/*===------------- avx512bwintrin.h - AVX512BW intrinsics ------------------===
 *
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
#error "Never use <avx512bwintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512BWINTRIN_H
#define __AVX512BWINTRIN_H

typedef unsigned int __mmask32;
typedef unsigned long long __mmask64;
typedef char __v64qi __attribute__ ((__vector_size__ (64)));
typedef short __v32hi __attribute__ ((__vector_size__ (64)));


/* Integer compare */

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpeq_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_pcmpeqb512_mask((__v64qi)__a, (__v64qi)__b,
                                                   (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpeq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_pcmpeqb512_mask((__v64qi)__a, (__v64qi)__b,
                                                   __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpeq_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 0,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpeq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 0,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpeq_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_pcmpeqw512_mask((__v32hi)__a, (__v32hi)__b,
                                                   (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpeq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_pcmpeqw512_mask((__v32hi)__a, (__v32hi)__b,
                                                   __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpeq_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 0,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpeq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 0,
                                                 __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpge_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 5,
                                                (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpge_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 5,
                                                __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpge_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 5,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpge_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 5,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpge_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 5,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpge_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 5,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpge_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 5,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpge_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 5,
                                                 __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpgt_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_pcmpgtb512_mask((__v64qi)__a, (__v64qi)__b,
                                                   (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpgt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_pcmpgtb512_mask((__v64qi)__a, (__v64qi)__b,
                                                   __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpgt_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 6,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpgt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 6,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpgt_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_pcmpgtw512_mask((__v32hi)__a, (__v32hi)__b,
                                                   (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpgt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_pcmpgtw512_mask((__v32hi)__a, (__v32hi)__b,
                                                   __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpgt_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 6,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpgt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 6,
                                                 __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmple_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 2,
                                                (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmple_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 2,
                                                __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmple_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 2,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmple_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 2,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmple_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 2,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmple_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 2,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmple_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 2,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmple_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 2,
                                                 __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmplt_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 1,
                                                (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmplt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 1,
                                                __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmplt_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 1,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmplt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 1,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmplt_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 1,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmplt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 1,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmplt_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 1,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmplt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 1,
                                                 __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpneq_epi8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 4,
                                                (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpneq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_cmpb512_mask((__v64qi)__a, (__v64qi)__b, 4,
                                                __u);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpneq_epu8_mask(__m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 4,
                                                 (__mmask64)-1);
}

static __inline__ __mmask64 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpneq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  return (__mmask64)__builtin_ia32_ucmpb512_mask((__v64qi)__a, (__v64qi)__b, 4,
                                                 __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpneq_epi16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 4,
                                                (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpneq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_cmpw512_mask((__v32hi)__a, (__v32hi)__b, 4,
                                                __u);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_cmpneq_epu16_mask(__m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 4,
                                                 (__mmask32)-1);
}

static __inline__ __mmask32 __attribute__((__always_inline__, __nodebug__))
_mm512_mask_cmpneq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  return (__mmask32)__builtin_ia32_ucmpw512_mask((__v32hi)__a, (__v32hi)__b, 4,
                                                 __u);
}

#define _mm512_cmp_epi8_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpb512_mask((__v64qi)(__m512i)(a), \
                                         (__v64qi)(__m512i)(b), \
                                         (p), (__mmask64)-1); })

#define _mm512_mask_cmp_epi8_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpb512_mask((__v64qi)(__m512i)(a), \
                                         (__v64qi)(__m512i)(b), \
                                         (p), (__mmask64)(m)); })

#define _mm512_cmp_epu8_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpb512_mask((__v64qi)(__m512i)(a), \
                                          (__v64qi)(__m512i)(b), \
                                          (p), (__mmask64)-1); })

#define _mm512_mask_cmp_epu8_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpb512_mask((__v64qi)(__m512i)(a), \
                                          (__v64qi)(__m512i)(b), \
                                          (p), (__mmask64)(m)); })

#define _mm512_cmp_epi16_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpw512_mask((__v32hi)(__m512i)(a), \
                                         (__v32hi)(__m512i)(b), \
                                         (p), (__mmask32)-1); })

#define _mm512_mask_cmp_epi16_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_cmpw512_mask((__v32hi)(__m512i)(a), \
                                         (__v32hi)(__m512i)(b), \
                                         (p), (__mmask32)(m)); })

#define _mm512_cmp_epu16_mask(a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpw512_mask((__v32hi)(__m512i)(a), \
                                          (__v32hi)(__m512i)(b), \
                                          (p), (__mmask32)-1); })

#define _mm512_mask_cmp_epu16_mask(m, a, b, p) __extension__ ({ \
  (__mmask16)__builtin_ia32_ucmpw512_mask((__v32hi)(__m512i)(a), \
                                          (__v32hi)(__m512i)(b), \
                                          (p), (__mmask32)(m)); })

#endif
