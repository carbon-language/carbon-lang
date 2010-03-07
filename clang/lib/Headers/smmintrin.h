/*===---- smmintrin.h - SSE intrinsics -------------------------------------===
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

#ifndef _SMMINTRIN_H
#define _SMMINTRIN_H

#ifndef __SSE4_1__
#error "SSE4.1 instruction set not enabled"
#else

#include <tmmintrin.h>

/* Type defines.  */
typedef double __v2df __attribute__ ((__vector_size__ (16)));
typedef long long __v2di __attribute__ ((__vector_size__ (16)));

/* SSE4 Rounding macros. */
#define _MM_FROUND_TO_NEAREST_INT    0x00
#define _MM_FROUND_TO_NEG_INF        0x01
#define _MM_FROUND_TO_POS_INF        0x02
#define _MM_FROUND_TO_ZERO           0x03
#define _MM_FROUND_CUR_DIRECTION     0x04

#define _MM_FROUND_RAISE_EXC         0x00
#define _MM_FROUND_NO_EXC            0x08

#define _MM_FROUND_NINT      (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEAREST_INT)
#define _MM_FROUND_FLOOR     (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_NEG_INF)
#define _MM_FROUND_CEIL      (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_POS_INF)
#define _MM_FROUND_TRUNC     (_MM_FROUND_RAISE_EXC | _MM_FROUND_TO_ZERO)
#define _MM_FROUND_RINT      (_MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION)
#define _MM_FROUND_NEARBYINT (_MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION)

#define _mm_ceil_ps(X)       _mm_round_ps((X), _MM_FROUND_CEIL)
#define _mm_ceil_pd(X)       _mm_round_pd((X), _MM_FROUND_CEIL)
#define _mm_ceil_ss(X, Y)    _mm_round_ss((X), (Y), _MM_FROUND_CEIL)
#define _mm_ceil_sd(X, Y)    _mm_round_sd((X), (Y), _MM_FROUND_CEIL)

#define _mm_floor_ps(X)      _mm_round_ps((X), _MM_FROUND_FLOOR)
#define _mm_floor_pd(X)      _mm_round_pd((X), _MM_FROUND_FLOOR)
#define _mm_floor_ss(X, Y)   _mm_round_ss((X), (Y), _MM_FROUND_FLOOR)
#define _mm_floor_sd(X, Y)   _mm_round_sd((X), (Y), _MM_FROUND_FLOOR)

#define _mm_round_ps(X, Y)      __builtin_ia32_roundps((X), (Y))
#define _mm_round_ss(X, Y, M)   __builtin_ia32_roundss((X), (Y), (M))
#define _mm_round_pd(X, M)      __builtin_ia32_roundpd((X), (M))
#define _mm_round_sd(X, Y, M)   __builtin_ia32_roundsd((X), (Y), (M))

/* SSE4 Packed Blending Intrinsics.  */
static inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_blend_pd (__m128d __V1, __m128d __V2, const int __M)
{
  return (__m128d) __builtin_ia32_blendpd ((__v2df)__V1, (__v2df)__V2, __M);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_blend_ps (__m128 __V1, __m128 __V2, const int __M)
{
  return (__m128) __builtin_ia32_blendps ((__v4sf)__V1, (__v4sf)__V2, __M);
}

static inline __m128d __attribute__((__always_inline__, __nodebug__))
_mm_blendv_pd (__m128d __V1, __m128d __V2, __m128d __M)
{
  return (__m128d) __builtin_ia32_blendvpd ((__v2df)__V1, (__v2df)__V2,
                                            (__v2df)__M);
}

static inline __m128 __attribute__((__always_inline__, __nodebug__))
_mm_blendv_ps (__m128 __V1, __m128 __V2, __m128 __M)
{
  return (__m128) __builtin_ia32_blendvps ((__v4sf)__V1, (__v4sf)__V2,
                                           (__v4sf)__M);
}

static inline __m128i __attribute__((__always_inline__, __nodebug__))
_mm_blendv_epi8 (__m128i __V1, __m128i __V2, __m128i __M)
{
  return (__m128i) __builtin_ia32_pblendvb128 ((__v16qi)__V1, (__v16qi)__V2,
                                               (__v16qi)__M);
}

static inline  __m128i __attribute__((__always_inline__, __nodebug__))
_mm_blend_epi16 (__m128i __V1, __m128i __V2, const int __M)
{
  return (__m128i) __builtin_ia32_pblendw128 ((__v8hi)__V1, (__v8hi)__V2, __M);
}

/* SSE4 Dword Multiply Instructions.  */
static inline  __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mullo_epi32 (__m128i __V1, __m128i __V2)
{
  return (__m128i) __builtin_ia32_pmulld128((__v4si)__V1, (__v4si)__V2);
}

static inline  __m128i __attribute__((__always_inline__, __nodebug__))
_mm_mul_epi32 (__m128i __V1, __m128i __V2)
{
  return (__m128i) __builtin_ia32_pmuldq128 ((__v4si)__V1, (__v4si)__V2);
}

/* SSE4 Floating Point Dot Product Instructions.  */
#define _mm_dp_ps(X, Y, M) __builtin_ia32_dpps ((X), (Y), (M))
#define _mm_dp_pd(X, Y, M) __builtin_ia32_dppd ((X), (Y), (M))

/* SSE4 Streaming Load Hint Instruction.  */
static inline  __m128i __attribute__((__always_inline__, __nodebug__))
_mm_stream_load_si128 (__m128i *__V)
{
  return (__m128i) __builtin_ia32_movntdqa ((__v2di *) __V);
}

#endif /* __SSE4_1__ */

#endif /* _SMMINTRIN_H */
