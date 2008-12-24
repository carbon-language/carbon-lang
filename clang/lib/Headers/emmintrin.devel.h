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
 
#ifndef __EMMINTRIN_H
#define __EMMINTRIN_H

#ifndef __SSE2__
#error "SSE2 instruction set not enabled"
#else

#include <xmmintrin.h>

typedef double __m128d __attribute__((__vector_size__(16)));
typedef long long __m128i __attribute__((__vector_size__(16)));

typedef int __v4si __attribute__((__vector_size__(16)));

static inline __m128d __attribute__((__always_inline__)) _mm_add_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_addsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_add_pd(__m128d a, __m128d b)
{
  return a + b;
}

static inline __m128d __attribute__((__always_inline__)) _mm_sub_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_subsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_sub_pd(__m128d a, __m128d b)
{
  return a - b;
}

static inline __m128d __attribute__((__always_inline__)) _mm_mul_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_mulsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_mul_pd(__m128d a, __m128d b)
{
  return a * b;
}

static inline __m128d __attribute__((__always_inline__)) _mm_div_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_divsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_div_pd(__m128d a, __m128d b)
{
  return a / b;
}

static inline __m128d __attribute__((__always_inline__)) _mm_sqrt_sd(__m128d a, __m128d b)
{
  __m128d c = __builtin_ia32_sqrtsd(b);
  return (__m128d) { c[0], a[1] };
}

static inline __m128d __attribute__((__always_inline__)) _mm_sqrt_pd(__m128d a)
{
  return __builtin_ia32_sqrtpd(a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_min_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_minsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_min_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_minpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_max_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_maxsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_max_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_maxpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_and_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_andpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_andnot_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_andnpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_or_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_orpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_xor_pd(__m128d a, __m128d b)
{
  return __builtin_ia32_xorpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpeq_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpeqpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmplt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpltpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmple_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmplepd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpgt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpltpd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpge_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmplepd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpord_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpordpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpunord_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpunordpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpneq_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpneqpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnlt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnltpd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnle_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnlepd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpngt_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnltpd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnge_pd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnlepd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpeq_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpeqsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmplt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpltsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmple_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmplesd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpgt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpltsd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpge_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmplesd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpord_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpordsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpunord_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpunordsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpneq_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpneqsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnlt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnltsd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnle_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnlesd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpngt_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnltsd(b, a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cmpnge_sd(__m128d a, __m128d b)
{
  return (__m128d)__builtin_ia32_cmpnlesd(b, a);
}

static inline int __attribute__((__always_inline__)) _mm_comieq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdeq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comilt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdlt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comile_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdle(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comigt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdgt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_comineq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_comisdneq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomieq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdeq(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomilt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdlt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomile_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdle(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomigt_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdgt(a, b);
}

static inline int __attribute__((__always_inline__)) _mm_ucomineq_sd(__m128d a, __m128d b)
{
  return __builtin_ia32_ucomisdneq(a, b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpd_ps(__m128d a)
{
  return __builtin_ia32_cvtpd2ps(a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cvtps_pd(__m128 a)
{
  return __builtin_ia32_cvtps2pd(a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cvtepi32_pd(__m128i a)
{
  return __builtin_ia32_cvtdq2pd((__v4si)a);
}

static inline __m128i __attribute__((__always_inline__)) _mm_cvtpd_epi32(__m128d a)
{
  return __builtin_ia32_cvtpd2dq(a);
}

static inline int __attribute__((__always_inline__)) _mm_cvtsd_si32(__m128d a)
{
  return __builtin_ia32_cvtsd2si(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtsd_ss(__m128 a, __m128d b)
{
  return __builtin_ia32_cvtsd2ss(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cvtsi32_sd(__m128d a, int b)
{
  return __builtin_ia32_cvtsi2sd(a, b);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cvtss_sd(__m128d a, __m128 b)
{
  return __builtin_ia32_cvtss2sd(a, b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_cvttpd_epi32(__m128d a)
{
  return (__m128i)__builtin_ia32_cvttpd2dq(a);
}

static inline int __attribute__((__always_inline__)) _mm_cvttsd_si32(__m128d a)
{
  return __builtin_ia32_cvttsd2si(a);
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvtpd_pi32(__m128d a)
{
  return (__m64)__builtin_ia32_cvtpd2pi(a);
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvttpd_pi32(__m128d a)
{
  return (__m64)__builtin_ia32_cvttpd2pi(a);
}

static inline __m128d __attribute__((__always_inline__)) _mm_cvtpi32_pd(__m64 a)
{
  return __builtin_ia32_cvtpi2pd((__v2si)a);
}

static inline double __attribute__((__always_inline__)) _mm_cvtsd_f64(__m128d a)
{
  return a[0];
}

#endif /* __SSE2__ */

#endif /* __EMMINTRIN_H */
