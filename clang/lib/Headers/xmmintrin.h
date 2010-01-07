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
#error "SSE instruction set not enabled"
#else

#include <mmintrin.h>

typedef float __v4sf __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));

#include <mm_malloc.h>

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_add_ss(__m128 a, __m128 b)
{
  a[0] += b[0];
  return a;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_add_ps(__m128 a, __m128 b)
{
  return a + b;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sub_ss(__m128 a, __m128 b)
{
  a[0] -= b[0];
  return a;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sub_ps(__m128 a, __m128 b)
{
  return a - b;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_mul_ss(__m128 a, __m128 b)
{
  a[0] *= b[0];
  return a;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_mul_ps(__m128 a, __m128 b)
{
  return a * b;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_div_ss(__m128 a, __m128 b)
{
  a[0] /= b[0];
  return a;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_div_ps(__m128 a, __m128 b)
{
  return a / b;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_ss(__m128 a)
{
  return __builtin_ia32_sqrtss(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_sqrt_ps(__m128 a)
{
  return __builtin_ia32_sqrtps(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rcp_ss(__m128 a)
{
  return __builtin_ia32_rcpss(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rcp_ps(__m128 a)
{
  return __builtin_ia32_rcpps(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rsqrt_ss(__m128 a)
{
  return __builtin_ia32_rsqrtss(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_rsqrt_ps(__m128 a)
{
  return __builtin_ia32_rsqrtps(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_min_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_minss(a, b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_min_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_minps(a, b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_max_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_maxss(a, b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_max_ps(__m128 a, __m128 b)
{
  return __builtin_ia32_maxps(a, b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_and_ps(__m128 a, __m128 b)
{
  typedef int __v4si __attribute__((__vector_size__(16)));
  return (__m128)((__v4si)a & (__v4si)b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_andnot_ps(__m128 a, __m128 b)
{
  typedef int __v4si __attribute__((__vector_size__(16)));
  return (__m128)(~(__v4si)a & (__v4si)b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_or_ps(__m128 a, __m128 b)
{
  typedef int __v4si __attribute__((__vector_size__(16)));
  return (__m128)((__v4si)a | (__v4si)b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_xor_ps(__m128 a, __m128 b)
{
  typedef int __v4si __attribute__((__vector_size__(16)));
  return (__m128)((__v4si)a ^ (__v4si)b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 0);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 0);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 1);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmplt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 1);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 2);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmple_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 2);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(b, a, 1);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(b, a, 1);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(b, a, 2);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpge_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(b, a, 2);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 4);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpneq_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 4);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnlt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 6);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnle_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 6);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(b, a, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpngt_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(b, a, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(b, a, 6);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpnge_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(b, a, 6);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 7);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpord_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 7);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_ss(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpss(a, b, 3);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cmpunord_ps(__m128 a, __m128 b)
{
  return (__m128)__builtin_ia32_cmpps(a, b, 3);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comieq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comieq(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comilt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comilt(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comile_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comile(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comigt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comigt(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comige_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comige(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_comineq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_comineq(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomieq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomieq(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomilt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomilt(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomile_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomile(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomigt_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomigt(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomige_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomige(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_ucomineq_ss(__m128 a, __m128 b)
{
  return __builtin_ia32_ucomineq(a, b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_si32(__m128 a)
{
  return __builtin_ia32_cvtss2si(a);
}

#ifdef __x86_64__

static inline long long __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_si64(__m128 a)
{
  return __builtin_ia32_cvtss2si64(a);
}

#endif

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi32(__m128 a)
{
  return (__m64)__builtin_ia32_cvtps2pi(a);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_cvttss_si32(__m128 a)
{
  return a[0];
}

static inline long long __attribute__((__always_inline__, __nodebug__))
_mm_cvttss_si64(__m128 a)
{
  return a[0];
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvttps_pi32(__m128 a)
{
  return (__m64)__builtin_ia32_cvttps2pi(a);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi32_ss(__m128 a, int b)
{
  a[0] = b;
  return a;
}

#ifdef __x86_64__

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_ss(__m128 a, long long b)
{
  a[0] = b;
  return a;
}

#endif

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi32_ps(__m128 a, __m64 b)
{
  return __builtin_ia32_cvtpi2ps(a, (__v2si)b);
}

static inline float __attribute__((__always_inline__, __nodebug__))
_mm_cvtss_f32(__m128 a)
{
  return a[0];
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadh_pi(__m128 a, __m64 const *p)
{
  __m128 b;
  b[0] = *(float*)p;
  b[1] = *((float*)p+1);
  return __builtin_shufflevector(a, b, 0, 1, 4, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadl_pi(__m128 a, __m64 const *p)
{
  __m128 b;
  b[0] = *(float*)p;
  b[1] = *((float*)p+1);
  return __builtin_shufflevector(a, b, 4, 5, 2, 3);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load_ss(float *p)
{
  return (__m128){ *p, 0, 0, 0 };
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load1_ps(float *p)
{
  return (__m128){ *p, *p, *p, *p };
}

#define        _mm_load_ps1(p) _mm_load1_ps(p)

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_load_ps(float *p)
{
  return *(__m128*)p;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadu_ps(float *p)
{
  return __builtin_ia32_loadups(p);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_loadr_ps(float *p)
{
  __m128 a = _mm_load_ps(p);
  return __builtin_shufflevector(a, a, 3, 2, 1, 0);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ss(float w)
{
  return (__m128){ w, 0, 0, 0 };
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set1_ps(float w)
{
  return (__m128){ w, w, w, w };
}

// Microsoft specific.
static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ps1(float w)
{
    return _mm_set1_ps(w);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_set_ps(float z, float y, float x, float w)
{
  return (__m128){ w, x, y, z };
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_setr_ps(float z, float y, float x, float w)
{
  return (__m128){ z, y, x, w };
}

static inline __m128 __attribute__((__always_inline__))
_mm_setzero_ps(void)
{
  return (__m128){ 0, 0, 0, 0 };
}

static inline void __attribute__((__always_inline__))
_mm_storeh_pi(__m64 *p, __m128 a)
{
  __builtin_ia32_storehps((__v2si *)p, a);
}

static inline void __attribute__((__always_inline__))
_mm_storel_pi(__m64 *p, __m128 a)
{
  __builtin_ia32_storelps((__v2si *)p, a);
}

static inline void __attribute__((__always_inline__))
_mm_store_ss(float *p, __m128 a)
{
  *p = a[0];
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_storeu_ps(float *p, __m128 a)
{
  __builtin_ia32_storeups(p, a);
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_store1_ps(float *p, __m128 a)
{
  a = __builtin_shufflevector(a, a, 0, 0, 0, 0);
  _mm_storeu_ps(p, a);
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_store_ps(float *p, __m128 a)
{
  *(__m128 *)p = a;
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_storer_ps(float *p, __m128 a)
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

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_stream_pi(__m64 *p, __m64 a)
{
  __builtin_ia32_movntq(p, a);
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_stream_ps(float *p, __m128 a)
{
  __builtin_ia32_movntps(p, a);
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_sfence(void)
{
  __builtin_ia32_sfence();
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_extract_pi16(__m64 a, int n)
{
  __v4hi b = (__v4hi)a;
  return (unsigned short)b[n & 3];
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_insert_pi16(__m64 a, int d, int n)
{
   __v4hi b = (__v4hi)a;
   b[n & 3] = d;
   return (__m64)b;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_max_pi16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmaxsw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_max_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmaxub((__v8qi)a, (__v8qi)b);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_min_pi16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pminsw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_min_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pminub((__v8qi)a, (__v8qi)b);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_pi8(__m64 a)
{
  return __builtin_ia32_pmovmskb((__v8qi)a);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mulhi_pu16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pmulhuw((__v4hi)a, (__v4hi)b);  
}

#define _mm_shuffle_pi16(a, n) \
  ((__m64)__builtin_shufflevector((__v4hi)(a), (__v4hi) {0}, \
                                  (n) & 0x3, ((n) & 0xc) >> 2, \
                                  ((n) & 0x30) >> 4, ((n) & 0xc0) >> 6))

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_maskmove_si64(__m64 d, __m64 n, char *p)
{
  __builtin_ia32_maskmovq((__v8qi)d, (__v8qi)n, p);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_avg_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pavgb((__v8qi)a, (__v8qi)b);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_avg_pu16(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_pavgw((__v4hi)a, (__v4hi)b);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sad_pu8(__m64 a, __m64 b)
{
  return (__m64)__builtin_ia32_psadbw((__v8qi)a, (__v8qi)b);
}

static inline unsigned int __attribute__((__always_inline__, __nodebug__))
_mm_getcsr(void)
{
  return __builtin_ia32_stmxcsr();
}

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_setcsr(unsigned int i)
{
  __builtin_ia32_ldmxcsr(i);
}

#define _mm_shuffle_ps(a, b, mask) \
        (__builtin_shufflevector(a, b, (mask) & 0x3, ((mask) & 0xc) >> 2, \
                                 (((mask) & 0x30) >> 4) + 4, \
                                 (((mask) & 0xc0) >> 6) + 4))

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 2, 6, 3, 7);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 0, 4, 1, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_move_ss(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 4, 1, 2, 3);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movehl_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 6, 7, 2, 3);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_movelh_ps(__m128 a, __m128 b)
{
  return __builtin_shufflevector(a, b, 0, 1, 4, 5);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi16_ps(__m64 a)
{
  __m64 b, c;
  __m128 r;

  b = _mm_setzero_si64();
  b = _mm_cmpgt_pi16(b, a);
  c = _mm_unpackhi_pi16(a, b);  
  r = _mm_setzero_ps();
  r = _mm_cvtpi32_ps(r, c);
  r = _mm_movelh_ps(r, r);
  c = _mm_unpacklo_pi16(a, b);  
  r = _mm_cvtpi32_ps(r, c);

  return r;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpu16_ps(__m64 a)
{
  __m64 b, c;
  __m128 r;

  b = _mm_setzero_si64();
  c = _mm_unpackhi_pi16(a, b);  
  r = _mm_setzero_ps();
  r = _mm_cvtpi32_ps(r, c);
  r = _mm_movelh_ps(r, r);
  c = _mm_unpacklo_pi16(a, b);  
  r = _mm_cvtpi32_ps(r, c);

  return r;
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi8_ps(__m64 a)
{
  __m64 b;
  
  b = _mm_setzero_si64();
  b = _mm_cmpgt_pi8(b, a);
  b = _mm_unpacklo_pi8(a, b);

  return _mm_cvtpi16_ps(b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpu8_ps(__m64 a)
{
  __m64 b;
  
  b = _mm_setzero_si64();
  b = _mm_unpacklo_pi8(a, b);

  return _mm_cvtpi16_ps(b);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_cvtpi32x2_ps(__m64 a, __m64 b)
{
  __m128 c;
  
  c = _mm_setzero_ps();  
  c = _mm_cvtpi32_ps(c, b);
  c = _mm_movelh_ps(c, c);

  return _mm_cvtpi32_ps(c, a);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi16(__m128 a)
{
  __m64 b, c;
  
  b = _mm_cvtps_pi32(a);
  a = _mm_movehl_ps(a, a);
  c = _mm_cvtps_pi32(a);
  
  return _mm_packs_pi16(b, c);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtps_pi8(__m128 a)
{
  __m64 b, c;
  
  b = _mm_cvtps_pi16(a);
  c = _mm_setzero_si64();
  
  return _mm_packs_pi16(b, c);
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_movemask_ps(__m128 a)
{
  return __builtin_ia32_movmskps(a);
}

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

#define _MM_EXCEPT_INVALID    (0x0001)
#define _MM_EXCEPT_DENORM     (0x0002)
#define _MM_EXCEPT_DIV_ZERO   (0x0004)
#define _MM_EXCEPT_OVERFLOW   (0x0008)
#define _MM_EXCEPT_UNDERFLOW  (0x0010)
#define _MM_EXCEPT_INEXACT    (0x0020)
#define _MM_EXCEPT_MASK       (0x003f)

#define _MM_MASK_INVALID      (0x0080)
#define _MM_MASK_DENORM       (0x0100)
#define _MM_MASK_DIV_ZERO     (0x0200)
#define _MM_MASK_OVERFLOW     (0x0400)
#define _MM_MASK_UNDERFLOW    (0x0800)
#define _MM_MASK_INEXACT      (0x1000)
#define _MM_MASK_MASK         (0x1f80)

#define _MM_ROUND_NEAREST     (0x0000)
#define _MM_ROUND_DOWN        (0x2000)
#define _MM_ROUND_UP          (0x4000)
#define _MM_ROUND_TOWARD_ZERO (0x6000)
#define _MM_ROUND_MASK        (0x6000)

#define _MM_FLUSH_ZERO_MASK   (0x8000)
#define _MM_FLUSH_ZERO_ON     (0x8000)
#define _MM_FLUSH_ZERO_OFF    (0x8000)

#define _MM_GET_EXCEPTION_MASK() (_mm_getcsr() & _MM_MASK_MASK)
#define _MM_GET_EXCEPTION_STATE() (_mm_getcsr() & _MM_EXCEPT_MASK)
#define _MM_GET_FLUSH_ZERO_MODE() (_mm_getcsr() & _MM_FLUSH_ZERO_MASK)
#define _MM_GET_ROUNDING_MODE() (_mm_getcsr() & _MM_ROUND_MASK)

#define _MM_SET_EXCEPTION_MASK(x) (_mm_setcsr((_mm_getcsr() & ~_MM_MASK_MASK) | (x)))
#define _MM_SET_EXCEPTION_STATE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_EXCEPT_MASK) | (x)))
#define _MM_SET_FLUSH_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_FLUSH_ZERO_MASK) | (x)))
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

/* Ugly hack for backwards-compatibility (compatible with gcc) */
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#endif /* __SSE__ */

#endif /* __XMMINTRIN_H */
