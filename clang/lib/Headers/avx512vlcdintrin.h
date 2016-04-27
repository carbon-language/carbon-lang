/*===---- avx512vlcdintrin.h - AVX512VL and AVX512CD intrinsics ---------------------------===
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
#error "Never use <avx512vlcdintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLCDINTRIN_H
#define __AVX512VLCDINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("avx512vl,avx512cd")))


static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_broadcastmb_epi64 (__mmask8 __A)
{
  return (__m128i) __builtin_ia32_broadcastmb128 (__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_broadcastmb_epi64 (__mmask8 __A)
{
  return (__m256i) __builtin_ia32_broadcastmb256 (__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_broadcastmw_epi32 (__mmask16 __A)
{
  return (__m128i) __builtin_ia32_broadcastmw128 (__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
_mm256_broadcastmw_epi32 (__mmask16 __A)
{
  return (__m256i) __builtin_ia32_broadcastmw256 (__A);
}


#undef __DEFAULT_FN_ATTRS

#endif /* __AVX512VLCDINTRIN_H */