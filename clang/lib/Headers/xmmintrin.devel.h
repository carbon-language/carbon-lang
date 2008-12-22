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

#include <mmintrin.h>

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

static inline int __attribute__((__always_inline__)) _mm_cvtss_si32(__m128 a)
{
  return __builtin_ia32_cvtss2si(a);
}

static inline long long __attribute__((__always_inline__)) _mm_cvtss_si64(__m128 a)
{
  return __builtin_ia32_cvtss2si64(a);
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvtps_pi32(__m128 a)
{
  return (__m64)__builtin_ia32_cvtps2pi(a);
}

static inline int __attribute__((__always_inline__)) _mm_cvttss_si32(__m128 a)
{
  return __builtin_ia32_cvttss2si(a);
}

static inline long long __attribute__((__always_inline__)) _mm_cvttss_si64(__m128 a)
{
  return __builtin_ia32_cvttss2si64(a);
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvttps_pi32(__m128 a)
{
  return (__m64)__builtin_ia32_cvttps2pi(a);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtsi32_ss(__m128 a, int b)
{
  return __builtin_ia32_cvtsi2ss(a, b);
}

#ifdef __x86_64__

static inline __m128 __attribute__((__always_inline__)) _mm_cvtsi64_ss(__m128 a, long long b)
{
  return __builtin_ia32_cvtsi642ss(a, b);
}

#endif

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpi32_ps(__m128 a, __m64 b)
{
  return __builtin_ia32_cvtpi2ps(a, (__v2si)b);
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpi16_ps(__m64 a)
{
  /* FIXME: Implement */
  return (__m128){ 0, 0, 0, 0 };
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpu16_ps(__m64 a)
{
  /* FIXME: Implement */
  return (__m128){ 0, 0, 0, 0 };  
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpi8_ps(__m64 a)
{
  /* FIXME: Implement */
  return (__m128){ 0, 0, 0, 0 };  
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpu8_ps(__m64 a)
{
  /* FIXME: Implement */
  return (__m128){ 0, 0, 0, 0 };  
}

static inline __m128 __attribute__((__always_inline__)) _mm_cvtpi32x2_ps(__m64 a, __m64 b)
{
  /* FIXME: Implement */
  return (__m128){ 0, 0, 0, 0 };  
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvtps_pi16(__m128 a)
{
  /* FIXME: Implement */
  return _mm_setzero_si64();
}

static inline __m64 __attribute__((__always_inline__)) _mm_cvtps_pi8(__m128 a)
{
  /* FIXME: Implement */
  return _mm_setzero_si64();
}

static inline float __attribute__((__always_inline__)) _mm_cvtss_f32(__m128 a)
{
  /* FIXME: Implement */
  return 0;
}

static inline __m128 __attribute__((__always_inline__)) _mm_loadh_pi(__m128 a, __m64 const *p)
{
  return __builtin_ia32_loadhps(a, (__v2si *)p);
}

static inline __m128 __attribute__((__always_inline__)) _mm_loadl_pi(__m128 a, __m64 const *p)
{
  return __builtin_ia32_loadlps(a, (__v2si *)p);
}

static inline __m128 __attribute__((__always_inline__)) _mm_load_ss(float *p)
{
  return (__m128){ *p, 0, 0, 0 };
}

static inline __m128 __attribute__((__always_inline__)) _mm_load1_ps(float *p)
{
  return (__m128){ *p, *p, *p, *p };
}

static inline __m128 __attribute__((__always_inline__)) _mm_load_ps(float *p)
{
  return *(__m128*)p;
}

static inline __m128 __attribute__((__always_inline__)) _mm_loadu_ps(float *p)
{
  return __builtin_ia32_loadups(p);
}

static inline __m128 __attribute__((__always_inline__)) _mm_loadr_ps(float *p)
{
  __m128 a = _mm_load_ps(p);
  return __builtin_shufflevector(a, a, 3, 2, 1, 0);
}

static inline __m128 __attribute__((__always_inline__)) _mm_set_ss(float w)
{
  return (__m128){ w, 0, 0, 0 };
}

static inline __m128 __attribute__((__always_inline__)) _mm_set1_ps(float w)
{
  return (__m128){ w, w, w, w };
}

static inline __m128 __attribute__((__always_inline__)) _mm_set_ps(float z, float y, float x, float w)
{
  return (__m128){ w, x, y, z };
}

static inline __m128 __attribute__((__always_inline__)) _mm_setr_ps(float z, float y, float x, float w)
{
  return (__m128){ z, y, x, w };
}

static inline __m128 __attribute__((__always__inline__)) _mm_setzero_ps(void)
{
  return (__m128){ 0, 0, 0, 0 };
}

static inline void __attribute__((__always__inline__)) _mm_storeh_pi(__m64 *p, __m128 a)
{
  __builtin_ia32_storehps((__v2si *)p, a);
}

static inline void __attribute__((__always__inline__)) _mm_storel_pi(__m64 *p, __m128 a)
{
  __builtin_ia32_storelps((__v2si *)p, a);
}

static inline void __attribute__((__always__inline__)) _mm_store_ss(float *p, __m128 a)
{
  *p = a[0];
}

static inline void __attribute__((__always_inline__)) _mm_storeu_ps(float *p, __m128 a)
{
  __builtin_ia32_storeups(p, a);
}

static inline void __attribute__((__always_inline__)) _mm_store1_ps(float *p, __m128 a)
{
  a = __builtin_shufflevector(a, a, 0, 0, 0, 0);
  _mm_storeu_ps(p, a);
}

static inline void __attribute__((__always_inline__)) _mm_store_ps(float *p, __m128 a)
{
  *(__m128 *)p = a;
}

static inline void __attribute__((__always_inline__)) _mm_storer_ps(float *p, __m128 a)
{
  a = __builtin_shufflevector(a, a, 3, 2, 1, 0);
  _mm_store_ps(p, a);
}

#define _MM_HINT_T0 1
#define _MM_HINT_T1 2
#define _MM_HINT_T2 3
#define _MM_HINT_NTA 0

/* FIXME: We have to #define this because "sel" must be a constant integer, and 
   Sema doesn't do any form of constant propagation yet. */

#define _mm_prefetch(a, sel) (__builtin_prefetch((void *)a, 0, sel))

static inline void __attribute__((__always_inline__)) _mm_stream_pi(__m64 *p, __m64 a)
{
  __builtin_ia32_movntq(p, a);
}

static inline void __attribute__((__always_inline__)) _mm_stream_ps(float *p, __m128 a)
{
  __builtin_ia32_movntps(p, a);
}

static inline void __attribute__((__always_inline__)) _mm_sfence(void)
{
  __builtin_ia32_sfence();
}

static inline int __attribute__((__always_inline__)) _mm_extract_pi16(__m64 a, int n)
{
  /* FIXME: 
   * This should force n to be an immediate.
   * This does not use the PEXTRW instruction. From looking at the LLVM source, the
     instruction doesn't seem to be hooked up. 
   * The code could probably be made better :)
   */
  __v4hi b = (__v4hi)a;
  return b[(n == 0) ? 0 : (n == 1 ? 1 : (n == 2 ? 2 : 3))];
}

static inline __m64 __attribute__((__always_inline__)) _mm_insert_pi16(__m64 a, int d, int n)
{
  /* FIXME: Implement this. We could add a __builtin_insertelement function that's similar to
     the already existing __builtin_shufflevector.
    */
   return (__m64){ 0LL };
}

static inline __m64 __attribute__((__always_inline__)) _mm_max_pi16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmaxsw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline__)) _mm_max_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmaxub((__v8qi)a, (__v8qi)b);
}

static inline __m64 __attribute__((__always_inline__)) _mm_min_pi16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pminsw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline__)) _mm_min_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pminub((__v8qi)a, (__v8qi)b);
}

static inline int __attribute__((__always_inline__)) _mm_movemask_pi8(__m64 a)
{
  return __builtin_ia32_pmovmskb((__v8qi)a);
}

static inline __m64 __attribute__((__always_inline__)) _mm_mulhi_pu16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmulhuw((__v4hi)a, (__v4hi)b);  
}

#define _mm_shuffle_pi16(a, n) ((__m64)__builtin_ia32_pshufw((__v4hi)a, n))

static inline void __attribute__((__always_inline__)) _mm_maskmove_si64(__m64 d, __m64 n, char *p)
{
  __builtin_ia32_maskmovq((__v8qi)d, (__v8qi)n, p);
}

static inline __m64 __attribute__((__always_inline__)) _mm_avg_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pavgb((__v8qi)a, (__v8qi)b);
}

static inline __m64 __attribute__((__always_inline__)) _mm_avg_pu16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pavgw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline___)) _mm_sad_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_psadbw((__v8qi)a, (__v8qi)b);
}

static inline unsigned int __attribute__((__always_inline___)) _mm_getcsr(void)
{
  return __builtin_ia32_stmxcsr();
}

static inline void __attribute__((__always_inline__)) _mm_setcsr(unsigned int i)
{
  __builtin_ia32_ldmxcsr(i);
}

#define _mm_shuffle_ps(a, b, mask) (__builtin_ia32_shufps(a, b, mask))

static inline __m128 __attribute__((__always_inline__)) _mm_unpackhi_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 2, 6, 3, 7);
}

static inline __m128 __attribute__((__always_inline__)) _mm_unpacklo_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 0, 4, 1, 5);
}

static inline __m128 __attribute__((__always_inline__)) _mm_move_ss(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 4, 1, 2, 3);
}

static inline __m128 __attribute__((__always_inline__)) _mm_movehl_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 6, 7, 2, 3);
}

static inline __m128 __attribute__((__always_inline__)) _mm_movelh_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 0, 1, 4, 5);
}

static inline int __attribute__((__always_inline__)) _mm_movemask_ps(__m128 a)
{
  return __builtin_ia32_movmskps(a);
}

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

#define _MM_MASK_MASK (0x1f80)
#define _MM_EXCEPT_MASK (0x003f)
#define _MM_FLUSH_MASK (0x8000)
#define _MM_ROUND_MASK (0x6000)

#define _MM_GET_EXCEPTION_MASK() (_mm_getcsr() & _MM_MASK_MASK)
#define _MM_GET_EXCEPTION_STATE() (_mm_getcsr() & _MM_EXCEPT_MASK)
#define _MM_GET_FLUSH_ZERO_MODE() (_mm_getcsr() & _MM_FLUSH_MASK)
#define _MM_GET_ROUNDING_MODE() (_mm_getcsr() & _MM_ROUND_MASK)

#define _MM_SET_EXCEPTION_MASK(x) (_mm_setcsr((_mm_getcsr() & ~_MM_FLUSH_MASK) | (x)))
#define _MM_SET_EXCEPTION_STATE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_EXCEPT_MASK) | (x)))
#define _MM_SET_FLUSH_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_FLUSH_MASK) | (x)))
#define _MM_SET_ROUNDING_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_ROUND_MASK) | (x)))

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movelh_ps(tmp3, tmp1); \
} while (0)

#endif /* __SSE__ */

#endif /* __XMMINTRIN_H */
