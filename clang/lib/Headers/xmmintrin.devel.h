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

static inline __m128 __attribute__((__always_inline__)) _mm_cmpeq_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpeqss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpeq_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpeqps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmplt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpltss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmplt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpltps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmple_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpless(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmple_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpleps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpgt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpltss(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpgt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpltps(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpge_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpless(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpge_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpleps(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpneq_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpneqss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpneq_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpneqps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnlt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnltss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnlt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnltps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnle_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnless(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnle_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnleps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpngt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnltss(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpngt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnltps(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnge_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnless(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpnge_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpnleps(b, a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpord_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpordss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpord_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpordps(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpunord_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpunordss(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cmpunord_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpunordps(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comieq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comieq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comilt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comilt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comile_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comile(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comigt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comigt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comige_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comige(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comineq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comineq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomieq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomieq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomilt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomilt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomile_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomile(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomigt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomigt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomige_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomige(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomineq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomineq(a, b);
}

#endif /* __SSE__ */

#endif /* __XMMINTRIN_H */
