/*===---- mmintrin.h - MMX intrinsics --------------------------------------===
 *
 * Copyright (c) 2008 Anders Carlsson
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

#ifndef __MMINTRIN_H
#define __MMINTRIN_H

#ifndef __MMX__
#error "MMX instruction set not enabled"
#else

typedef long long __m64 __attribute__((vector_size(8)));

typedef int __v2si __attribute__((vector_size(8)));
typedef short __v4hi __attribute__((vector_size(8)));
typedef char __v8qi __attribute__((vector_size(8)));

inline void __attribute__((__always_inline__)) _mm_empty()
{
    __builtin_ia32_emms();
}

inline __m64 __attribute__((__always_inline__)) _mm_cvtsi32_si64(int i)
{
    return (__m64)(__v2si){i, 0};
}

inline int __attribute__((__always_inline__)) _mm_cvtsi64_si32(__m64 m)
{
    __v2si __mmx_var2 = (__v2si)m;
    return __mmx_var2[0];
}

inline __m64 __attribute__((__always_inline__)) _mm_cvtsi64_m64(long long i)
{
    return (__m64)i;
}

inline long long __attribute__((__always_inline__)) _mm_cvtm64_si64(__m64 m)
{
    return (long long)m;
}

inline __m64 __attribute__((__always_inline__)) _mm_packs_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_packsswb((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_packs_pi32(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_packssdw((__v2si)m1, (__v2si)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_packs_pu16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_packuswb((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpackhi_pi8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v8qi)m1, (__v8qi)m2, 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpackhi_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v4hi)m1, (__v4hi)m2, 2, 4+2, 3, 4+3);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpackhi_pi32(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v2si)m1, (__v2si)m2, 1, 2+1);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpacklo_pi8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v8qi)m1, (__v8qi)m2, 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpacklo_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v4hi)m1, (__v4hi)m2, 0, 4+0, 1, 4+1);
}

inline __m64 __attribute__((__always_inline__)) _mm_unpacklo_pi32(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_shufflevector((__v2si)m1, (__v2si)m2, 0, 2+0);
}

inline __m64 __attribute__((__always_inline__)) _mm_add_pi8(__m64 m1, __m64 m2)
{
    return (__m64)((__v8qi)m1 + (__v8qi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_add_pi16(__m64 m1, __m64 m2)
{
    return (__m64)((__v4hi)m1 + (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_add_pi32(__m64 m1, __m64 m2)
{
    return (__m64)((__v2si)m1 + (__v2si)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_adds_pi8(__m64 m1, __m64 m2) 
{
    return (__m64)__builtin_ia32_paddsb((__v8qi)m1, (__v8qi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_adds_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_paddsw((__v4hi)m1, (__v4hi)m2);    
}

inline __m64 __attribute__((__always_inline__)) _mm_adds_pu8(__m64 m1, __m64 m2) 
{
    return (__m64)__builtin_ia32_paddusb((__v8qi)m1, (__v8qi)m2);
}
 
inline __m64 __attribute__((__always_inline__)) _mm_adds_pu16(__m64 m1, __m64 m2) 
{
    return (__m64)__builtin_ia32_paddusw((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_sub_pi8(__m64 m1, __m64 m2)
{
    return (__m64)((__v8qi)m1 - (__v8qi)m2);
}
 
inline __m64 __attribute__((__always_inline__)) _mm_sub_pi16(__m64 m1, __m64 m2)
{
    return (__m64)((__v4hi)m1 - (__v4hi)m2);
}
 
inline __m64 __attribute__((__always_inline__)) _mm_sub_pi32(__m64 m1, __m64 m2)
{
    return (__m64)((__v2si)m1 - (__v2si)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_subs_pi8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_psubsb((__v8qi)m1, (__v8qi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_subs_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_psubsw((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_subs_pu8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_psubusb((__v8qi)m1, (__v8qi)m2);
}
 
inline __m64 __attribute__((__always_inline__)) _mm_subs_pu16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_psubusw((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_madd_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pmaddwd((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_mulhi_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pmulhw((__v4hi)m1, (__v4hi)m2);
}
 
inline __m64 __attribute__((__always_inline__)) _mm_mullo_pi16(__m64 m1, __m64 m2) 
{
    return (__m64)((__v4hi)m1 * (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_sll_pi16(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psllw((__v4hi)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_slli_pi16(__m64 m, int count)
{
    return (__m64)__builtin_ia32_psllwi((__v4hi)m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_sll_pi32(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_pslld((__v2si)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_slli_pi32(__m64 m, int count)
{
    return (__m64)__builtin_ia32_pslldi((__v2si)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_sll_pi64(__m64 m, __m64 count)
{
    return __builtin_ia32_psllq(m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_slli_pi64(__m64 m, int count)
{
    return __builtin_ia32_psllqi(m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_sra_pi16(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psraw((__v4hi)m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_srai_pi16(__m64 m, int count)
{
    return (__m64)__builtin_ia32_psrawi((__v4hi)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_sra_pi32(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psrad((__v2si)m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_srai_pi32(__m64 m, int count)
{
    return (__m64)__builtin_ia32_psradi((__v2si)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_srl_pi16(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psrlw((__v4hi)m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_srli_pi16(__m64 m, int count)
{
    return (__m64)__builtin_ia32_psrlwi((__v4hi)m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_srl_pi32(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psrld((__v2si)m, count);       
}

inline __m64 __attribute__((__always_inline__)) _mm_srli_pi32(__m64 m, int count)
{
    return (__m64)__builtin_ia32_psrldi((__v2si)m, count);
}

inline __m64 __attribute__((__always_inline__)) _mm_srl_pi64(__m64 m, __m64 count)
{
    return (__m64)__builtin_ia32_psrlq(m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_srli_pi64(__m64 m, int count)
{
    return __builtin_ia32_psrlqi(m, count);    
}

inline __m64 __attribute__((__always_inline__)) _mm_and_si64(__m64 m1, __m64 m2)
{
    return m1 & m2;
}

inline __m64 __attribute__((__always_inline__)) _mm_andnot_si64(__m64 m1, __m64 m2)
{
    return ~m1 & m2;
}

inline __m64 __attribute__((__always_inline__)) _mm_or_si64(__m64 m1, __m64 m2)
{
    return m1 | m2;
}

inline __m64 __attribute__((__always_inline__)) _mm_xor_si64(__m64 m1, __m64 m2)
{
    return m1 ^ m2;
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpeq_pi8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpeqb((__v8qi)m1, (__v8qi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpeq_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpeqw((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpeq_pi32(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpeqd((__v2si)m1, (__v2si)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpgt_pi8(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpgtb((__v8qi)m1, (__v8qi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpgt_pi16(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpgtw((__v4hi)m1, (__v4hi)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_cmpgt_pi32(__m64 m1, __m64 m2)
{
    return (__m64)__builtin_ia32_pcmpgtd((__v2si)m1, (__v2si)m2);
}

inline __m64 __attribute__((__always_inline__)) _mm_setzero_si64()
{
    return (__m64){ 0LL };
}

inline __m64 __attribute__((__always_inline__)) _mm_set_pi32(int i1, int i0)
{
    return (__m64)(__v2si){ i0, i1 };
}

inline __m64 __attribute__((__always_inline__)) _mm_set_pi16(short s3, short s2, short s1, short s0)
{
    return (__m64)(__v4hi){ s0, s1, s2, s3 };    
}

inline __m64 __attribute__((__always_inline__)) _mm_set_pi8(char b7, char b6, char b5, char b4, char b3, char b2, char b1, char b0)
{
    return (__m64)(__v8qi){ b0, b1, b2, b3, b4, b5, b6, b7 };
}

inline __m64 __attribute__((__always_inline__)) _mm_set1_pi32(int i)
{
    return (__m64)(__v2si){ i, i };
}

inline __m64 __attribute__((__always_inline__)) _mm_set1_pi16(short s)
{
    return (__m64)(__v4hi){ s };
}

inline __m64 __attribute__((__always_inline__)) _mm_set1_pi8(char b)
{
    return (__m64)(__v8qi){ b };
}

inline __m64 __attribute__((__always_inline__)) _mm_setr_pi32(int i1, int i0)
{
    return (__m64)(__v2si){ i1, i0 };
}

inline __m64 __attribute__((__always_inline__)) _mm_setr_pi16(short s3, short s2, short s1, short s0)
{
    return (__m64)(__v4hi){ s3, s2, s1, s0 };
}

inline __m64 __attribute__((__always_inline__)) _mm_setr_pi8(char b7, char b6, char b5, char b4, char b3, char b2, char b1, char b0)
{
    return (__m64)(__v8qi){ b7, b6, b5, b4, b3, b2, b1, b0 };
}

#endif /* __MMX__ */

#endif /* __MMINTRIN_H */

