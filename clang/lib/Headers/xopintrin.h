/*===---- xopintrin.h - FMA4 intrinsics ------------------------------------===
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

#ifndef __X86INTRIN_H
#error "Never use <fma4intrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __XOPINTRIN_H
#define __XOPINTRIN_H

#ifndef __XOP__
# error "XOP instruction set is not enabled"
#else

#include <fma4intrin.h>

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccs_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssww((__v8hi)__A, (__v8hi)__B, (__v8hi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macc_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsww((__v8hi)__A, (__v8hi)__B, (__v8hi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccsd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccs_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdd((__v4si)__A, (__v4si)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macc_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdd((__v4si)__A, (__v4si)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccslo_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdql((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macclo_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdql((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maccshi_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacssdqh((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_macchi_epi32(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmacsdqh((__v4si)__A, (__v4si)__B, (__v2di)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maddsd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmadcsswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_maddd_epi16(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpmadcswd((__v8hi)__A, (__v8hi)__B, (__v4si)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddw_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbd((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddbq((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddwq((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epi32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadddq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddw_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubd((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddubq((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddd_epu16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadduwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphadduwq((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_haddq_epu32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphaddudq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubw_epi8(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubbw((__v16qi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubd_epi16(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubwd((__v8hi)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_hsubq_epi32(__m128i __A)
{
  return (__m128i)__builtin_ia32_vphsubdq((__v4si)__A);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_cmov_si128(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpcmov(__A, __B, __C);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmov_si256(__m256i __A, __m256i __B, __m256i __C)
{
  return (__m256i)__builtin_ia32_vpcmov_256(__A, __B, __C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_perm_epi8(__m128i __A, __m128i __B, __m128i __C)
{
  return (__m128i)__builtin_ia32_vpperm((__v16qi)__A, (__v16qi)__B, (__v16qi)__C);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi8(__m128i __A, __m128 __B)
{
  return (__m128i)__builtin_ia32_vprotb((__v16qi)__A, (__v16qi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi16(__m128i __A, __m128 __B)
{
  return (__m128i)__builtin_ia32_vprotw((__v8hi)__A, (__v8hi)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi32(__m128i __A, __m128 __B)
{
  return (__m128i)__builtin_ia32_vprotd((__v4si)__A, (__v4si)__B);
}

static __inline__ __m128i __attribute__((__always_inline__, __nodebug__))
_mm_rot_epi64(__m128i __A, __m128 __B)
{
  return (__m128i)__builtin_ia32_vprotq((__v2di)__A, (__v2di)__B);
}

#define _mm_roti_epi8(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotbi((__v16qi)__A, (N)); })

#define _mm_roti_epi16(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotwi((__v8hi)__A, (N)); })

#define _mm_roti_epi32(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotdi((__v4si)__A, (N)); })

#define _mm_roti_epi64(A, N) __extension__ ({ \
  __m128i __A = (A); \
  (__m128i)__builtin_ia32_vprotqi((__v2di)__A, (N)); })

#endif /* __XOP__ */

#endif /* __XOPINTRIN_H */
