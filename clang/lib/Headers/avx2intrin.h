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
_mm256_avg_epu8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pavgb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_avg_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pavgw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_blendv_epi8(__m256i __V1, __m256i __V2, __m256i __M)
{
  return (__m256i)__builtin_ia32_pblendvb256((__v32qi)__V1, (__v32qi)__V2,
                                              (__v32qi)__M);
}

#define _mm256_blend_epi16(V1, V2, M) __extension__ ({ \
  __m256i __V1 = (V1); \
  __m256i __V2 = (V2); \
  (__m256i)__builtin_ia32_pblendw256((__v16hi)__V1, (__v16hi)__V2, M); })

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi8(__m256i a, __m256i b)
{
  return (__m256i)((__v32qi)a == (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi16(__m256i a, __m256i b)
{
  return (__m256i)((__v16hi)a == (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi32(__m256i a, __m256i b)
{
  return (__m256i)((__v8si)a == (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpeq_epi64(__m256i a, __m256i b)
{
  return (__m256i)((__v4di)a == (__v4di)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi8(__m256i a, __m256i b)
{
  return (__m256i)((__v32qi)a > (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi16(__m256i a, __m256i b)
{
  return (__m256i)((__v16hi)a > (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi32(__m256i a, __m256i b)
{
  return (__m256i)((__v8si)a > (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cmpgt_epi64(__m256i a, __m256i b)
{
  return (__m256i)((__v4di)a > (__v4di)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_epi16(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phaddw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadd_epi32(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phaddd256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hadds_epi16(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phaddsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_epi16(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phsubw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsub_epi32(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phsubd256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_hsubs_epi16(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_phsubsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_maddubs_epi16(__m256i a, __m256i b)
{
    return (__m256i)__builtin_ia32_pmaddubsw256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_madd_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaddwd256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxsb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epi32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxsd256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxub256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxuw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_max_epu32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmaxud256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminsb256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epi32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminsd256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu8(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminub256((__v32qi)a, (__v32qi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminuw256 ((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_min_epu32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pminud256((__v8si)a, (__v8si)b);
}

static __inline__ int __attribute__((__always_inline__, __nodebug__))
_mm256_movemask_epi8(__m256i a)
{
  return __builtin_ia32_pmovmskb256((__v32qi)a);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi16(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbw256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbd256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi8_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxbq256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi16_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxwd256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi16_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxwq256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepi32_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovsxdq256((__v4si)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi16(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbw256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbd256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu8_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxbq256((__v16qi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu16_epi32(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxwd256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu16_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxwq256((__v8hi)__V);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_cvtepu32_epi64(__m128i __V)
{
  return (__m256i)__builtin_ia32_pmovzxdq256((__v4si)__V);
}

static __inline__  __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mul_epi32(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmuldq256((__v8si)a, (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhrs_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmulhrsw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhi_epu16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmulhuw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mulhi_epi16(__m256i a, __m256i b)
{
  return (__m256i)__builtin_ia32_pmulhw256((__v16hi)a, (__v16hi)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mullo_epi16(__m256i a, __m256i b)
{
  return (__m256i)((__v16hi)a * (__v16hi)b);
}

static __inline__  __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mullo_epi32 (__m256i a, __m256i b)
{
  return (__m256i)((__v8si)a * (__v8si)b);
}

static __inline__ __m256i __attribute__((__always_inline__, __nodebug__))
_mm256_mul_epu32(__m256i a, __m256i b)
{
  return __builtin_ia32_pmuludq256((__v8si)a, (__v8si)b);
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
