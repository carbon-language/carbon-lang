/*===---- avx2intrin.h - AVX2 intrinsics -----------------------------------===
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
#error "Never use <avx2intrin.h> directly; include <immintrin.h> instead."
#endif

/* SSE4 Multiple Packed Sums of Absolute Difference.  */
#define _mm256_mpsadbw_epu8(X, Y, M) __builtin_ia32_mpsadbw256((X), (Y), (M))

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi8(__m256i a)
{
    return (__m256i)__builtin_ia32_pabsb256((__v32qi)a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi16(__m256i a)
{
    return (__m256i)__builtin_ia32_pabsw256((__v16hi)a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_abs_epi32(__m256i a)
{
    return (__m256i)__builtin_ia32_pabsd256((__v8si)a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packs_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_packsswb256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packs_epi32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_packssdw256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packus_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_packuswb256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_packus_epi32(__m256i __V1, __m256i __V2)
{
  return (__m256i) __builtin_ia32_packusdw256((__v8si)__V1, (__v8si)__V2);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi8(__m256i a, __m256i b)
{
  return (__m256i)((__v32qi)a + (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi16(__m256i a, __m256i b)
{
  return (__m256i)((__v16hi)a + (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi32(__m256i a, __m256i b)
{
  return (__m256i)((__v8si)a + (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_add_epi64(__m256i a, __m256i b)
{
  return a + b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epi8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_paddsb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_paddsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epu8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_paddusb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_adds_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_paddusw256((__v16hi)a, (__v16hi)b);
}

#define _mm256_alignr_epi8(a, b, n) __extension__ ({ \
  __m256i __a = (a); \
  __m256i __b = (b); \
  (__m256i)__builtin_ia32_palignr256((__v32qi)__a, (__v32qi)__b, (n)); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_and_si256(__m256i a, __m256i b)
{
  return a & b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_andnot_si256(__m256i a, __m256i b)
{
  return ~a & b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_or_si256(__m256i a, __m256i b)
{
  return a | b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi8(__m256i a, __m256i b)
{
  return (__m256i)((__v32qi)a - (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi16(__m256i a, __m256i b)
{
  return (__m256i)((__v16hi)a - (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi32(__m256i a, __m256i b)
{
  return (__m256i)((__v8si)a - (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_sub_epi64(__m256i a, __m256i b)
{
  return a - b;
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epi8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_psubsb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_psubsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epu8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_psubusb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_subs_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_psubusw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_xor_si256(__m256i a, __m256i b)
{
  return a ^ b;
}
