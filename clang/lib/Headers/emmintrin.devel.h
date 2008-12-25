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
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef char __v16qi __attribute__((__vector_size__(16)));

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

static inline __m128d __attribute__((__always_inline__)) _mm_load_pd(double const *dp)
{
  return *(__m128d*)dp;
}

static inline __m128d __attribute__((__always_inline__)) _mm_load1_pd(double const *dp)
{
  return (__m128d){ dp[0], dp[0] };
}

static inline __m128d __attribute__((__always_inline__)) _mm_loadr_pd(double const *dp)
{
  return (__m128d){ dp[1], dp[0] };
}

static inline __m128d __attribute__((__always_inline__)) _mm_loadu_pd(double const *dp)
{
  return __builtin_ia32_loadupd(dp);
}

static inline __m128d __attribute__((__always_inline__)) _mm_load_sd(double const *dp)
{
  return (__m128d){ *dp, 0.0 };
}

static inline __m128d __attribute__((__always_inline__)) _mm_loadh_pd(__m128d a, double const *dp)
{
  return __builtin_shufflevector(a, *(__m128d *)dp, 0, 2);
}

static inline __m128d __attribute__((__always_inline__)) _mm_loadl_pd(__m128d a, double const *dp)
{
  return __builtin_shufflevector(a, *(__m128d *)dp, 2, 1);
}

static inline __m128d __attribute__((__always_inline__)) _mm_set_sd(double w)
{
  return (__m128d){ w, 0 };
}

static inline __m128d __attribute__((__always_inline__)) _mm_set1_pd(double w)
{
  return (__m128d){ w, w };
}

static inline __m128d __attribute__((__always_inline__)) _mm_set_pd(double w, double x)
{
  return (__m128d){ w, x };
}

static inline __m128d __attribute__((__always_inline__)) _mm_setr_pd(double w, double x)
{
  return (__m128d){ x, w };
}

static inline __m128d __attribute__((__always_inline__)) _mm_setzero_pd(void)
{
  return (__m128d){ 0, 0 };
}

static inline __m128d __attribute__((__always_inline__)) _mm_move_sd(__m128d a, __m128d b)
{
  return (__m128d){ b[0], a[1] };
}

static inline void __attribute__((__always_inline__)) _mm_store_sd(double *dp, __m128d a)
{
  dp[0] = a[0];
}

static inline void __attribute__((__always_inline__)) _mm_store1_pd(double *dp, __m128d a)
{
  dp[0] = a[0];
  dp[1] = a[0];
}

static inline void __attribute__((__always_inline__)) _mm_store_pd(double *dp, __m128d a)
{
  *(__m128d *)dp = a;
}

static inline void __attribute__((__always_inline__)) _mm_storeu_pd(double *dp, __m128d a)
{
  __builtin_ia32_storeupd(dp, a);
}

static inline void __attribute__((__always_inline__)) _mm_storer_pd(double *dp, __m128d a)
{
  dp[0] = a[1];
  dp[1] = a[0];
}

static inline void __attribute__((__always_inline__)) _mm_storeh_pd(double *dp, __m128d a)
{
  dp[0] = a[1];
}

static inline void __attribute__((__always_inline__)) _mm_storel_pd(double *dp, __m128d a)
{
  dp[0] = a[0];
}

static inline __m128i __attribute__((__always_inline__)) _mm_add_epi8(__m128i a, __m128i b)
{
  return (__m128i)((__v16qi)a + (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_add_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a + (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_add_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a + (__v4si)b);
}

static inline __m64 __attribute__((__always_inline__)) _mm_add_si64(__m64 a, __m64 b)
{
  return a + b;
}

static inline __m128i __attribute__((__always_inline__)) _mm_add_epi64(__m128i a, __m128i b)
{
  return a + b;
}

static inline __m128i __attribute__((__always_inline__)) _mm_adds_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddsb128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_adds_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddsw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_adds_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddusb128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_adds_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_paddusw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_avg_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pavgb128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_avg_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pavgw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_madd_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaddwd128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_max_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaxsw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_max_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmaxub128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_min_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pminsw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_min_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pminub128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_mulhi_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmulhw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_mulhi_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmulhuw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_mullo_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_pmullw128((__v8hi)a, (__v8hi)b);
}

__m64 _mm_mul_su32(__m64 a, __m64 b)
{
  return __builtin_ia32_pmuludq((__v2si)a, (__v2si)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_mul_epu32(__m128i a, __m128i b)
{
  return __builtin_ia32_pmuludq128((__v4si)a, (__v4si)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_sad_epu(__m128i a, __m128i b)
{
  return __builtin_ia32_psadbw128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_sub_epi8(__m128i a, __m128i b)
{
  return (__m128i)((__v16qi)a - (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_sub_epi16(__m128i a, __m128i b)
{
  return (__m128i)((__v8hi)a - (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_sub_epi32(__m128i a, __m128i b)
{
  return (__m128i)((__v4si)a - (__v4si)b);
}

__m64 _mm_sub_si64(__m64 a, __m64 b)
{
  return a - b;
}

static inline __m128i __attribute__((__always_inline__)) _mm_sub_epi64(__m128i a, __m128i b)
{
  return a - b;
}

static inline __m128i __attribute__((__always_inline__)) _mm_subs_epi8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubsb128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_subs_epi16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubsw128((__v8hi)a, (__v8hi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_subs_epu8(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubusb128((__v16qi)a, (__v16qi)b);
}

static inline __m128i __attribute__((__always_inline__)) _mm_subs_epu16(__m128i a, __m128i b)
{
  return (__m128i)__builtin_ia32_psubusw128((__v8hi)a, (__v8hi)b);
}

#endif /* __SSE2__ */

#endif /* __EMMINTRIN_H */
