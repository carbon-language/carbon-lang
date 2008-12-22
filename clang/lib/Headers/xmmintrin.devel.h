/*===---- xmmintrin.h - SSE intrinsics -------------------------------------===
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
 
#ifndef __XMMINTRIN_H
#define __XMMINTRIN_H
 
#ifndef __SSE__
#error "MMX instruction set not enabled"
#else

typedef float __m128 __attribute__((__vector_size__(16)));

static inline __m128 __attribute__((__always_inline__)) _mm_add_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_addss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_add_ps(__m128 a, __m128 b)
{
  return a + b;
}

static inline __m128 __attribute__((__always_inline__)) _mm_sub_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_subss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_sub_ps(__m128 a, __m128 b)
{
  return a - b;
}

static inline __m128 __attribute__((__always_inline__)) _mm_mul_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_mulss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_mul_ps(__m128 a, __m128 b)
{
  return a * b;
}

static inline __m128 __attribute__((__always_inline__)) _mm_div_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_divss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_div_ps(__m128 a, __m128 b)
{
  return a / b;
}

static inline __m128 __attribute__((__always_inline__)) _mm_sqrt_ss(__m128 a)
{
  return __builtin_ia32_sqrtss(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_sqrt_ps(__m128 a)
{
  return __builtin_ia32_sqrtps(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_rcp_ss(__m128 a)
{
  return __builtin_ia32_rcpss(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_rcp_ps(__m128 a)
{
  return __builtin_ia32_rcpps(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_rsqrt_ss(__m128 a)
{
  return __builtin_ia32_rsqrtss(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_rsqrt_ps(__m128 a)
{
  return __builtin_ia32_rsqrtps(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_min_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_minss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_min_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_minps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_max_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_maxss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_max_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_maxps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_and_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_andps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_andnot_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_andnps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_or_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_orps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_xor_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_xorps(a, b);
}

#endif /* __SSE__ */

#endif /* __XMMINTRIN_H */
