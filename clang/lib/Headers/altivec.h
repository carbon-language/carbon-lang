/*===---- altivec.h - Standard header for type generic math ---------------===*\
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
\*===----------------------------------------------------------------------===*/

#ifndef __ALTIVEC_H
#define __ALTIVEC_H

#ifndef __ALTIVEC__
#error "AltiVec support not enabled"
#endif

/* constants for mapping CR6 bits to predicate result. */

#define __CR6_EQ     0
#define __CR6_EQ_REV 1
#define __CR6_LT     2
#define __CR6_LT_REV 3

#define _ATTRS_o_ai __attribute__((__overloadable__, __always_inline__))

/* vec_abs */

#define __builtin_vec_abs vec_abs
#define __builtin_altivec_abs_v16qi vec_abs
#define __builtin_altivec_abs_v8hi  vec_abs
#define __builtin_altivec_abs_v4si  vec_abs

static vector signed char _ATTRS_o_ai
vec_abs(vector signed char a)
{
  return __builtin_altivec_vmaxsb(a, -a);
}

static vector signed short _ATTRS_o_ai
vec_abs(vector signed short a)
{
  return __builtin_altivec_vmaxsh(a, -a);
}

static vector signed int _ATTRS_o_ai
vec_abs(vector signed int a)
{
  return __builtin_altivec_vmaxsw(a, -a);
}

static vector float _ATTRS_o_ai
vec_abs(vector float a)
{
  vector unsigned int res = (vector unsigned int)a &
                            (vector unsigned int)(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
  return (vector float)res;
}

/* vec_abss */

#define __builtin_vec_abss vec_abss
#define __builtin_altivec_abss_v16qi vec_abss
#define __builtin_altivec_abss_v8hi  vec_abss
#define __builtin_altivec_abss_v4si  vec_abss

static vector signed char _ATTRS_o_ai
vec_abss(vector signed char a)
{
  return __builtin_altivec_vmaxsb(a, __builtin_altivec_vsubsbs(
    (vector signed char)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), a));
}

static vector signed short _ATTRS_o_ai
vec_abss(vector signed short a)
{
  return __builtin_altivec_vmaxsh(a, __builtin_altivec_vsubshs(
    (vector signed short)(0, 0, 0, 0, 0, 0, 0, 0), a));
}

static vector signed int _ATTRS_o_ai
vec_abss(vector signed int a)
{
  return __builtin_altivec_vmaxsw(a, __builtin_altivec_vsubsws(
    (vector signed int)(0, 0, 0, 0), a));
}

/* vec_add */

#define __builtin_altivec_vaddubm vec_add
#define __builtin_altivec_vadduhm vec_add
#define __builtin_altivec_vadduwm vec_add
#define __builtin_altivec_vaddfp  vec_add
#define __builtin_vec_vaddubm vec_add
#define __builtin_vec_vadduhm vec_add
#define __builtin_vec_vadduwm vec_add
#define __builtin_vec_vaddfp  vec_add
#define vec_vaddubm vec_add
#define vec_vadduhm vec_add
#define vec_vadduwm vec_add
#define vec_vaddfp  vec_add

static vector signed char _ATTRS_o_ai
vec_add(vector signed char a, vector signed char b)
{
  return a + b;
}

static vector unsigned char _ATTRS_o_ai
vec_add(vector unsigned char a, vector unsigned char b)
{
  return a + b;
}

static vector short _ATTRS_o_ai
vec_add(vector short a, vector short b)
{
  return a + b;
}

static vector unsigned short _ATTRS_o_ai
vec_add(vector unsigned short a, vector unsigned short b)
{
  return a + b;
}

static vector int _ATTRS_o_ai
vec_add(vector int a, vector int b)
{
  return a + b;
}

static vector unsigned int _ATTRS_o_ai
vec_add(vector unsigned int a, vector unsigned int b)
{
  return a + b;
}

static vector float _ATTRS_o_ai
vec_add(vector float a, vector float b)
{
  return a + b;
}

/* vec_addc */

#define __builtin_vec_addc __builtin_altivec_vaddcuw
#define vec_vaddcuw        __builtin_altivec_vaddcuw
#define vec_addc           __builtin_altivec_vaddcuw

/* vec_adds */

#define __builtin_vec_vaddsbs __builtin_altivec_vaddsbs
#define __builtin_vec_vaddubs __builtin_altivec_vaddubs
#define __builtin_vec_vaddshs __builtin_altivec_vaddshs
#define __builtin_vec_vadduhs __builtin_altivec_vadduhs
#define __builtin_vec_vaddsws __builtin_altivec_vaddsws
#define __builtin_vec_vadduws __builtin_altivec_vadduws
#define vec_vaddsbs __builtin_altivec_vaddsbs
#define vec_vaddubs __builtin_altivec_vaddubs
#define vec_vaddshs __builtin_altivec_vaddshs
#define vec_vadduhs __builtin_altivec_vadduhs
#define vec_vaddsws __builtin_altivec_vaddsws
#define vec_vadduws __builtin_altivec_vadduws

static vector signed char _ATTRS_o_ai
vec_adds(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vaddsbs(a, b);
}

static vector unsigned char _ATTRS_o_ai
vec_adds(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vaddubs(a, b);
}

static vector short _ATTRS_o_ai
vec_adds(vector short a, vector short b)
{
  return __builtin_altivec_vaddshs(a, b);
}

static vector unsigned short _ATTRS_o_ai
vec_adds(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vadduhs(a, b);
}

static vector int _ATTRS_o_ai
vec_adds(vector int a, vector int b)
{
  return __builtin_altivec_vaddsws(a, b);
}

static vector unsigned int _ATTRS_o_ai
vec_adds(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vadduws(a, b);
}

/* vec_sub */

#define __builtin_altivec_vsububm vec_sub
#define __builtin_altivec_vsubuhm vec_sub
#define __builtin_altivec_vsubuwm vec_sub
#define __builtin_altivec_vsubfp  vec_sub
#define __builtin_vec_vsububm vec_sub
#define __builtin_vec_vsubuhm vec_sub
#define __builtin_vec_vsubuwm vec_sub
#define __builtin_vec_vsubfp  vec_sub
#define vec_vsububm vec_sub
#define vec_vsubuhm vec_sub
#define vec_vsubuwm vec_sub
#define vec_vsubfp  vec_sub

static vector signed char _ATTRS_o_ai
vec_sub(vector signed char a, vector signed char b)
{
  return a - b;
}

static vector unsigned char _ATTRS_o_ai
vec_sub(vector unsigned char a, vector unsigned char b)
{
  return a - b;
}

static vector short _ATTRS_o_ai
vec_sub(vector short a, vector short b)
{
  return a - b;
}

static vector unsigned short _ATTRS_o_ai
vec_sub(vector unsigned short a, vector unsigned short b)
{
  return a - b;
}

static vector int _ATTRS_o_ai
vec_sub(vector int a, vector int b)
{
  return a - b;
}

static vector unsigned int _ATTRS_o_ai
vec_sub(vector unsigned int a, vector unsigned int b)
{
  return a - b;
}

static vector float _ATTRS_o_ai
vec_sub(vector float a, vector float b)
{
  return a - b;
}

/* vec_subs */

#define __builtin_vec_vsubsbs __builtin_altivec_vsubsbs
#define __builtin_vec_vsububs __builtin_altivec_vsububs
#define __builtin_vec_vsubshs __builtin_altivec_vsubshs
#define __builtin_vec_vsubuhs __builtin_altivec_vsubuhs
#define __builtin_vec_vsubsws __builtin_altivec_vsubsws
#define __builtin_vec_vsubuws __builtin_altivec_vsubuws
#define vec_vsubsbs __builtin_altivec_vsubsbs
#define vec_vsububs __builtin_altivec_vsububs
#define vec_vsubshs __builtin_altivec_vsubshs
#define vec_vsubuhs __builtin_altivec_vsubuhs
#define vec_vsubsws __builtin_altivec_vsubsws
#define vec_vsubuws __builtin_altivec_vsubuws

static vector signed char _ATTRS_o_ai
vec_subs(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vsubsbs(a, b);
}

static vector unsigned char _ATTRS_o_ai
vec_subs(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vsububs(a, b);
}

static vector short _ATTRS_o_ai
vec_subs(vector short a, vector short b)
{
  return __builtin_altivec_vsubshs(a, b);
}

static vector unsigned short _ATTRS_o_ai
vec_subs(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vsubuhs(a, b);
}

static vector int _ATTRS_o_ai
vec_subs(vector int a, vector int b)
{
  return __builtin_altivec_vsubsws(a, b);
}

static vector unsigned int _ATTRS_o_ai
vec_subs(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vsubuws(a, b);
}

/* vec_avg */

#define __builtin_vec_vavgsb __builtin_altivec_vavgsb
#define __builtin_vec_vavgub __builtin_altivec_vavgub
#define __builtin_vec_vavgsh __builtin_altivec_vavgsh
#define __builtin_vec_vavguh __builtin_altivec_vavguh
#define __builtin_vec_vavgsw __builtin_altivec_vavgsw
#define __builtin_vec_vavguw __builtin_altivec_vavguw
#define vec_vavgsb __builtin_altivec_vavgsb
#define vec_vavgub __builtin_altivec_vavgub
#define vec_vavgsh __builtin_altivec_vavgsh
#define vec_vavguh __builtin_altivec_vavguh
#define vec_vavgsw __builtin_altivec_vavgsw
#define vec_vavguw __builtin_altivec_vavguw

static vector signed char _ATTRS_o_ai
vec_avg(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vavgsb(a, b);
}

static vector unsigned char _ATTRS_o_ai
vec_avg(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vavgub(a, b);
}

static vector short _ATTRS_o_ai
vec_avg(vector short a, vector short b)
{
  return __builtin_altivec_vavgsh(a, b);
}

static vector unsigned short _ATTRS_o_ai
vec_avg(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vavguh(a, b);
}

static vector int _ATTRS_o_ai
vec_avg(vector int a, vector int b)
{
  return __builtin_altivec_vavgsw(a, b);
}

static vector unsigned int _ATTRS_o_ai
vec_avg(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vavguw(a, b);
}

/* vec_st */

#define __builtin_vec_st vec_st
#define vec_stvx         vec_st

static void _ATTRS_o_ai
vec_st(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvx(a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_st(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvx((vector int)a, b, (void *)c);
}

/* vec_stl */

#define __builtin_vec_stl vec_stl
#define vec_stvxl         vec_stl

static void _ATTRS_o_ai
vec_stl(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvxl(a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_stl(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvxl((vector int)a, b, (void *)c);
}

/* vec_ste */

#define __builtin_vec_stvebx __builtin_altivec_stvebx
#define __builtin_vec_stvehx __builtin_altivec_stvehx
#define __builtin_vec_stvewx __builtin_altivec_stvewx
#define vec_stvebx __builtin_altivec_stvebx
#define vec_stvehx __builtin_altivec_stvehx
#define vec_stvewx __builtin_altivec_stvewx

static void _ATTRS_o_ai
vec_ste(vector signed char a, int b, vector signed char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector unsigned char a, int b, vector unsigned char *c)
{
  __builtin_altivec_stvebx((vector char)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector short a, int b, vector short *c)
{
  __builtin_altivec_stvehx(a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector unsigned short a, int b, vector unsigned short *c)
{
  __builtin_altivec_stvehx((vector short)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector int a, int b, vector int *c)
{
  __builtin_altivec_stvewx(a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector unsigned int a, int b, vector unsigned int *c)
{
  __builtin_altivec_stvewx((vector int)a, b, (void *)c);
}

static void _ATTRS_o_ai
vec_ste(vector float a, int b, vector float *c)
{
  __builtin_altivec_stvewx((vector int)a, b, (void *)c);
}

/* vec_cmpb */

#define vec_cmpb           __builtin_altivec_vcmpbfp
#define vec_vcmpbfp        __builtin_altivec_vcmpbfp
#define __builtin_vec_cmpb __builtin_altivec_vcmpbfp

/* vec_cmpeq */

#define __builtin_vec_cmpeq vec_cmpeq

static vector /*bool*/ char _ATTRS_o_ai
vec_cmpeq(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb((vector char)a, (vector char)b);
}

static vector /*bool*/ char _ATTRS_o_ai
vec_cmpeq(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb((vector char)a, (vector char)b);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmpeq(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh(a, b);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmpeq(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh((vector short)a, (vector short)b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpeq(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw(a, b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpeq(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw((vector int)a, (vector int)b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpeq(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp(a, b);
}

/* vec_cmpge */

#define vec_cmpge           __builtin_altivec_vcmpgefp
#define vec_vcmpgefp        __builtin_altivec_vcmpgefp
#define __builtin_vec_cmpge __builtin_altivec_vcmpgefp

/* vec_cmpgt */

#define vec_vcmpgtsb __builtin_altivec_vcmpgtsb
#define vec_vcmpgtub __builtin_altivec_vcmpgtub
#define vec_vcmpgtsh __builtin_altivec_vcmpgtsh
#define vec_vcmpgtuh __builtin_altivec_vcmpgtuh
#define vec_vcmpgtsw __builtin_altivec_vcmpgtsw
#define vec_vcmpgtuw __builtin_altivec_vcmpgtuw
#define vec_vcmpgtfp __builtin_altivec_vcmpgtfp
#define __builtin_vec_vcmpgtsb __builtin_altivec_vcmpgtsb
#define __builtin_vec_vcmpgtub __builtin_altivec_vcmpgtub
#define __builtin_vec_vcmpgtsh __builtin_altivec_vcmpgtsh
#define __builtin_vec_vcmpgtuh __builtin_altivec_vcmpgtuh
#define __builtin_vec_vcmpgtsw __builtin_altivec_vcmpgtsw
#define __builtin_vec_vcmpgtuw __builtin_altivec_vcmpgtuw
#define __builtin_vec_vcmpgtfp __builtin_altivec_vcmpgtfp

static vector /*bool*/ char _ATTRS_o_ai
vec_cmpgt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb(a, b);
}

static vector /*bool*/ char _ATTRS_o_ai
vec_cmpgt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub(a, b);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmpgt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh(a, b);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmpgt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh(a, b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpgt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw(a, b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpgt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw(a, b);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmpgt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp(a, b);
}

/* vec_cmple */

#define __builtin_vec_cmple vec_cmple

static vector /*bool*/ int __attribute__((__always_inline__))
vec_cmple(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp(b, a);
}

/* vec_cmplt */

#define __builtin_vec_cmplt vec_cmplt

static vector /*bool*/ char _ATTRS_o_ai
vec_cmplt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb(b, a);
}

static vector /*bool*/ char _ATTRS_o_ai
vec_cmplt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub(b, a);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmplt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh(b, a);
}

static vector /*bool*/ short _ATTRS_o_ai
vec_cmplt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh(b, a);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmplt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw(b, a);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmplt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw(b, a);
}

static vector /*bool*/ int _ATTRS_o_ai
vec_cmplt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp(b, a);
}

/* vec_max */

#define __builtin_vec_vmaxsb __builtin_altivec_vmaxsb
#define __builtin_vec_vmaxub __builtin_altivec_vmaxub
#define __builtin_vec_vmaxsh __builtin_altivec_vmaxsh
#define __builtin_vec_vmaxuh __builtin_altivec_vmaxuh
#define __builtin_vec_vmaxsw __builtin_altivec_vmaxsw
#define __builtin_vec_vmaxuw __builtin_altivec_vmaxuw
#define __builtin_vec_vmaxfp __builtin_altivec_vmaxfp
#define vec_vmaxsb __builtin_altivec_vmaxsb
#define vec_vmaxub __builtin_altivec_vmaxub
#define vec_vmaxsh __builtin_altivec_vmaxsh
#define vec_vmaxuh __builtin_altivec_vmaxuh
#define vec_vmaxsw __builtin_altivec_vmaxsw
#define vec_vmaxuw __builtin_altivec_vmaxuw
#define vec_vmaxfp __builtin_altivec_vmaxfp
#define __builtin_vec_max vec_max

static vector signed char _ATTRS_o_ai
vec_max(vector signed  char a, vector signed char b)
{
  return __builtin_altivec_vmaxsb(a, b);
}

static vector unsigned char _ATTRS_o_ai
vec_max(vector unsigned  char a, vector unsigned char b)
{
  return __builtin_altivec_vmaxub(a, b);
}

static vector short _ATTRS_o_ai
vec_max(vector short a, vector short b)
{
  return __builtin_altivec_vmaxsh(a, b);
}

static vector unsigned short _ATTRS_o_ai
vec_max(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vmaxuh(a, b);
}

static vector int _ATTRS_o_ai
vec_max(vector int a, vector int b)
{
  return __builtin_altivec_vmaxsw(a, b);
}

static vector unsigned int _ATTRS_o_ai
vec_max(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vmaxuw(a, b);
}

static vector float _ATTRS_o_ai
vec_max(vector float a, vector float b)
{
  return __builtin_altivec_vmaxfp(a, b);
}

/* vec_mfvscr */

#define __builtin_vec_mfvscr __builtin_altivec_mfvscr
#define vec_mfvscr           __builtin_altivec_mfvscr

/* vec_min */

#define __builtin_vec_vminsb __builtin_altivec_vminsb
#define __builtin_vec_vminub __builtin_altivec_vminub
#define __builtin_vec_vminsh __builtin_altivec_vminsh
#define __builtin_vec_vminuh __builtin_altivec_vminuh
#define __builtin_vec_vminsw __builtin_altivec_vminsw
#define __builtin_vec_vminuw __builtin_altivec_vminuw
#define __builtin_vec_vminfp __builtin_altivec_vminfp
#define vec_vminsb __builtin_altivec_vminsb
#define vec_vminub __builtin_altivec_vminub
#define vec_vminsh __builtin_altivec_vminsh
#define vec_vminuh __builtin_altivec_vminuh
#define vec_vminsw __builtin_altivec_vminsw
#define vec_vminuw __builtin_altivec_vminuw
#define vec_vminfp __builtin_altivec_vminfp
#define __builtin_vec_min vec_min

static vector signed char _ATTRS_o_ai
vec_min(vector signed  char a, vector signed char b)
{
  return __builtin_altivec_vminsb(a, b);
}

static vector unsigned char _ATTRS_o_ai
vec_min(vector unsigned  char a, vector unsigned char b)
{
  return __builtin_altivec_vminub(a, b);
}

static vector short _ATTRS_o_ai
vec_min(vector short a, vector short b)
{
  return __builtin_altivec_vminsh(a, b);
}

static vector unsigned short _ATTRS_o_ai
vec_min(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vminuh(a, b);
}

static vector int _ATTRS_o_ai
vec_min(vector int a, vector int b)
{
  return __builtin_altivec_vminsw(a, b);
}

static vector unsigned int _ATTRS_o_ai
vec_min(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vminuw(a, b);
}

static vector float _ATTRS_o_ai
vec_min(vector float a, vector float b)
{
  return __builtin_altivec_vminfp(a, b);
}

/* vec_mtvscr */

#define __builtin_vec_mtvscr __builtin_altivec_mtvscr
#define vec_mtvscr           __builtin_altivec_mtvscr

/* ------------------------------ predicates ------------------------------------ */

static int __attribute__((__always_inline__))
__builtin_vec_vcmpeq_p(char CR6_param, vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(CR6_param, a, b);
}

static int __attribute__((__always_inline__))
__builtin_vec_vcmpge_p(char CR6_param, vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(CR6_param, a, b);
}

static int __attribute__((__always_inline__))
__builtin_vec_vcmpgt_p(char CR6_param, vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(CR6_param, a, b);
}

/* vec_all_eq */

static int _ATTRS_o_ai
vec_all_eq(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_all_eq(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_all_eq(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_eq(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT, (vector short)a, (vector short)b);
}

static int _ATTRS_o_ai
vec_all_eq(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_eq(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT, (vector int)a, (vector int)b);
}

static int _ATTRS_o_ai
vec_all_eq(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, a, b);
}

/* vec_all_ge */

static int _ATTRS_o_ai
vec_all_ge(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_ge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, b, a);
}

/* vec_all_gt */

static int _ATTRS_o_ai
vec_all_gt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, a, b);
}

static int _ATTRS_o_ai
vec_all_gt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, a, b);
}

/* vec_all_in */

static int __attribute__((__always_inline__))
vec_all_in(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ, a, b);
}

/* vec_all_le */

static int _ATTRS_o_ai
vec_all_le(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_le(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, a, b);
}

/* vec_all_lt */

static int _ATTRS_o_ai
vec_all_lt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT, b, a);
}

static int _ATTRS_o_ai
vec_all_lt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT, b, a);
}

/* vec_all_nan */

static int __attribute__((__always_inline__))
vec_all_nan(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, a, a);
}

/* vec_all_ne */

static int _ATTRS_o_ai
vec_all_ne(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_all_ne(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_all_ne(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_ne(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ, (vector short)a, (vector short)b);
}

static int _ATTRS_o_ai
vec_all_ne(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, a, b);
}

static int _ATTRS_o_ai
vec_all_ne(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ, (vector int)a, (vector int)b);
}

static int _ATTRS_o_ai
vec_all_ne(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ, a, b);
}

/* vec_all_nge */

static int __attribute__((__always_inline__))
vec_all_nge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, a, b);
}

/* vec_all_ngt */

static int __attribute__((__always_inline__))
vec_all_ngt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, a, b);
}

/* vec_all_nle */

static int __attribute__((__always_inline__))
vec_all_nle(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_EQ, b, a);
}

/* vec_all_nlt */

static int __attribute__((__always_inline__))
vec_all_nlt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ, b, a);
}

/* vec_all_numeric */

static int __attribute__((__always_inline__))
vec_all_numeric(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT, a, a);
}

/* vec_any_eq */

static int _ATTRS_o_ai
vec_any_eq(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_any_eq(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_EQ_REV, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_any_eq(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_eq(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_EQ_REV, (vector short)a, (vector short)b);
}

static int _ATTRS_o_ai
vec_any_eq(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_eq(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_EQ_REV, (vector int)a, (vector int)b);
}

static int _ATTRS_o_ai
vec_any_eq(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, a, b);
}

/* vec_any_ge */

static int _ATTRS_o_ai
vec_any_ge(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_ge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, b, a);
}

/* vec_any_gt */

static int _ATTRS_o_ai
vec_any_gt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_gt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, a, b);
}

/* vec_any_le */

static int _ATTRS_o_ai
vec_any_le(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_le(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, a, b);
}

/* vec_any_lt */

static int _ATTRS_o_ai
vec_any_lt(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpgtsb_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpgtub_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector short a, vector short b)
{
  return __builtin_altivec_vcmpgtsh_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpgtuh_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector int a, vector int b)
{
  return __builtin_altivec_vcmpgtsw_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpgtuw_p(__CR6_EQ_REV, b, a);
}

static int _ATTRS_o_ai
vec_any_lt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_EQ_REV, b, a);
}

/* vec_any_nan */

static int __attribute__((__always_inline__))
vec_any_nan(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, a, a);
}

/* vec_any_ne */

static int _ATTRS_o_ai
vec_any_ne(vector signed char a, vector signed char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_any_ne(vector unsigned char a, vector unsigned char b)
{
  return __builtin_altivec_vcmpequb_p(__CR6_LT_REV, (vector char)a, (vector char)b);
}

static int _ATTRS_o_ai
vec_any_ne(vector short a, vector short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_ne(vector unsigned short a, vector unsigned short b)
{
  return __builtin_altivec_vcmpequh_p(__CR6_LT_REV, (vector short)a, (vector short)b);
}

static int _ATTRS_o_ai
vec_any_ne(vector int a, vector int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, a, b);
}

static int _ATTRS_o_ai
vec_any_ne(vector unsigned int a, vector unsigned int b)
{
  return __builtin_altivec_vcmpequw_p(__CR6_LT_REV, (vector int)a, (vector int)b);
}

static int _ATTRS_o_ai
vec_any_ne(vector float a, vector float b)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_LT_REV, a, b);
}

/* vec_any_nge */

static int __attribute__((__always_inline__))
vec_any_nge(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, a, b);
}

/* vec_any_ngt */

static int __attribute__((__always_inline__))
vec_any_ngt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, a, b);
}

/* vec_any_nle */

static int __attribute__((__always_inline__))
vec_any_nle(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgefp_p(__CR6_LT_REV, b, a);
}

/* vec_any_nlt */

static int __attribute__((__always_inline__))
vec_any_nlt(vector float a, vector float b)
{
  return __builtin_altivec_vcmpgtfp_p(__CR6_LT_REV, b, a);
}

/* vec_any_numeric */

static int __attribute__((__always_inline__))
vec_any_numeric(vector float a)
{
  return __builtin_altivec_vcmpeqfp_p(__CR6_EQ_REV, a, a);
}

/* vec_any_out */

static int __attribute__((__always_inline__))
vec_any_out(vector float a, vector float b)
{
  return __builtin_altivec_vcmpbfp_p(__CR6_EQ_REV, a, b);
}

#undef _ATTRS_o_ai

#endif /* __ALTIVEC_H */
