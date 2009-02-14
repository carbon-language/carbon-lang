/*===---- mmintrin.h - MMX intrinsics --------------------------------------===
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

typedef long long __m64 __attribute__((__vector_size__(8)));

typedef int __v2si __attribute__((__vector_size__(8)));
typedef short __v4hi __attribute__((__vector_size__(8)));
typedef char __v8qi __attribute__((__vector_size__(8)));

static inline void __attribute__((__always_inline__, __nodebug__))
_mm_empty(void)
{
    __builtin_ia32_emms();
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi32_si64(int __i)
{
    return (__m64)(__v2si){__i, 0};
}

static inline int __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_si32(__m64 __m)
{
    __v2si __mmx_var2 = (__v2si)__m;
    return __mmx_var2[0];
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cvtsi64_m64(long long __i)
{
    return (__m64)__i;
}

static inline long long __attribute__((__always_inline__, __nodebug__))
_mm_cvtm64_si64(__m64 __m)
{
    return (long long)__m;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_packs_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_packsswb((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_packs_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_packssdw((__v2si)__m1, (__v2si)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_packs_pu16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_packuswb((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v8qi)__m1, (__v8qi)__m2, 4, 8+4, 5,
                                          8+5, 6, 8+6, 7, 8+7);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v4hi)__m1, (__v4hi)__m2, 2, 4+2, 3,
                                          4+3);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpackhi_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v2si)__m1, (__v2si)__m2, 1, 2+1);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v8qi)__m1, (__v8qi)__m2, 0, 8+0, 1,
                                          8+1, 2, 8+2, 3, 8+3);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v4hi)__m1, (__v4hi)__m2, 0, 4+0, 1,
                                          4+1);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_unpacklo_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_shufflevector((__v2si)__m1, (__v2si)__m2, 0, 2+0);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_add_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v8qi)__m1 + (__v8qi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_add_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v4hi)__m1 + (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_add_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v2si)__m1 + (__v2si)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_adds_pi8(__m64 __m1, __m64 __m2) 
{
    return (__m64)__builtin_ia32_paddsb((__v8qi)__m1, (__v8qi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_adds_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_paddsw((__v4hi)__m1, (__v4hi)__m2);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_adds_pu8(__m64 __m1, __m64 __m2) 
{
    return (__m64)__builtin_ia32_paddusb((__v8qi)__m1, (__v8qi)__m2);
}
 
static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_adds_pu16(__m64 __m1, __m64 __m2) 
{
    return (__m64)__builtin_ia32_paddusw((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sub_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v8qi)__m1 - (__v8qi)__m2);
}
 
static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sub_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v4hi)__m1 - (__v4hi)__m2);
}
 
static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sub_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)((__v2si)__m1 - (__v2si)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_subs_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_psubsb((__v8qi)__m1, (__v8qi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_subs_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_psubsw((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_subs_pu8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_psubusb((__v8qi)__m1, (__v8qi)__m2);
}
 
static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_subs_pu16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_psubusw((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_madd_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pmaddwd((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mulhi_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pmulhw((__v4hi)__m1, (__v4hi)__m2);
}
 
static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_mullo_pi16(__m64 __m1, __m64 __m2) 
{
    return (__m64)((__v4hi)__m1 * (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sll_pi16(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psllw((__v4hi)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_slli_pi16(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_psllwi((__v4hi)__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sll_pi32(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_pslld((__v2si)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_slli_pi32(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_pslldi((__v2si)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sll_si64(__m64 __m, __m64 __count)
{
    return __builtin_ia32_psllq(__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_slli_si64(__m64 __m, int __count)
{
    return __builtin_ia32_psllqi(__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sra_pi16(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psraw((__v4hi)__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srai_pi16(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_psrawi((__v4hi)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_sra_pi32(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psrad((__v2si)__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srai_pi32(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_psradi((__v2si)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srl_pi16(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psrlw((__v4hi)__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srli_pi16(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_psrlwi((__v4hi)__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srl_pi32(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psrld((__v2si)__m, __count);       
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srli_pi32(__m64 __m, int __count)
{
    return (__m64)__builtin_ia32_psrldi((__v2si)__m, __count);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srl_si64(__m64 __m, __m64 __count)
{
    return (__m64)__builtin_ia32_psrlq(__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_srli_si64(__m64 __m, int __count)
{
    return __builtin_ia32_psrlqi(__m, __count);    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_and_si64(__m64 __m1, __m64 __m2)
{
    return __m1 & __m2;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_andnot_si64(__m64 __m1, __m64 __m2)
{
    return ~__m1 & __m2;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_or_si64(__m64 __m1, __m64 __m2)
{
    return __m1 | __m2;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_xor_si64(__m64 __m1, __m64 __m2)
{
    return __m1 ^ __m2;
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpeqb((__v8qi)__m1, (__v8qi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpeqw((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpeq_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpeqd((__v2si)__m1, (__v2si)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_pi8(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpgtb((__v8qi)__m1, (__v8qi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_pi16(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpgtw((__v4hi)__m1, (__v4hi)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_cmpgt_pi32(__m64 __m1, __m64 __m2)
{
    return (__m64)__builtin_ia32_pcmpgtd((__v2si)__m1, (__v2si)__m2);
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_setzero_si64(void)
{
    return (__m64){ 0LL };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set_pi32(int __i1, int __i0)
{
    return (__m64)(__v2si){ __i0, __i1 };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set_pi16(short __s3, short __s2, short __s1, short __s0)
{
    return (__m64)(__v4hi){ __s0, __s1, __s2, __s3 };    
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set_pi8(char __b7, char __b6, char __b5, char __b4, char __b3, char __b2,
            char __b1, char __b0)
{
    return (__m64)(__v8qi){ __b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7 };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set1_pi32(int __i)
{
    return (__m64)(__v2si){ __i, __i };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set1_pi16(short __s)
{
    return (__m64)(__v4hi){ __s };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_set1_pi8(char __b)
{
    return (__m64)(__v8qi){ __b };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_setr_pi32(int __i1, int __i0)
{
    return (__m64)(__v2si){ __i1, __i0 };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_setr_pi16(short __s3, short __s2, short __s1, short __s0)
{
    return (__m64)(__v4hi){ __s3, __s2, __s1, __s0 };
}

static inline __m64 __attribute__((__always_inline__, __nodebug__))
_mm_setr_pi8(char __b7, char __b6, char __b5, char __b4, char __b3, char __b2,
             char __b1, char __b0)
{
    return (__m64)(__v8qi){ __b7, __b6, __b5, __b4, __b3, __b2, __b1, __b0 };
}

#endif /* __MMX__ */

#endif /* __MMINTRIN_H */

