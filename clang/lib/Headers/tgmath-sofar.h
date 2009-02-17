/*===---- tgmath.h - Standard header for type generic math ----------------===*\
 *
 * Copyright (c) 2009 Chris Lattner
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

#ifndef __TGMATH_H
#define __TGMATH_H

/* C99 7.22 Type-generic math <tgmath.h>. */
#include <math.h>

/* C++ handles type genericity with overloading in math.h. */
#ifndef __cplusplus
#include <complex.h>

#define __TG_UNARY_OVERLOAD(DSTTYPE, TYPE, SRCFN, DSTFN) \
  static DSTTYPE __attribute__((overloadable, always_inline)) __tg_ ## SRCFN(TYPE x) { return DSTFN(x); }


/* __TG_RC_1 - Unary functions defined on both real and complex values. */
#define __TG_RC_1(op, REALFN, COMPLEXFN) \
  __TG_UNARY_OVERLOAD(float, float, REALFN, REALFN ## f)                     \
  __TG_UNARY_OVERLOAD(double, double, REALFN, REALFN)                         \
  __TG_UNARY_OVERLOAD(long double, long double, REALFN, REALFN ## l)               \
  __TG_UNARY_OVERLOAD(double, long long, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(double, long, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(double, int, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(double, unsigned long long, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(double, unsigned long, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(double, unsigned int, REALFN, REALFN)                  \
  __TG_UNARY_OVERLOAD(_Complex float, _Complex float, REALFN, COMPLEXFN ## f)         \
  __TG_UNARY_OVERLOAD(_Complex double, _Complex double, REALFN, COMPLEXFN)             \
  __TG_UNARY_OVERLOAD(_Complex long double, _Complex long double, REALFN, COMPLEXFN ## l)

/* C99 7.22p4, functions in both math.h and complex.h. */
__TG_RC_1(x, acos, cacos)
#define acos(x) __tg_acos(x)
__TG_RC_1(x, asin, casin)
#define asin(x) __tg_asin(x)


#define atan(x) \
  __builtin_overload(1, x, catanl, catan, catanf, atanl, atan, atanf)
#define acosh(x) \
  __builtin_overload(1, x, cacoshl, cacosh, cacoshf, acoshl, acosh, acoshf)
#define asinh(x) \
  __builtin_overload(1, x, casinhl, casinh, casinhf, asinhl, asinh, asinhf)
#define atanh(x) \
  __builtin_overload(1, x, catanhl, catanh, catanhf, atanhl, atanh, atanhf)
#define cos(x) \
  __builtin_overload(1, x, ccosl, ccos, ccosf, cosl, cos, cosf)
#define sin(x) \
  __builtin_overload(1, x, csinl, csin, csinf, sinl, sin, sinf)
#define tan(x) \
  __builtin_overload(1, x, ctanl, ctan, ctanf, tanl, tan, tanf)
#define cosh(x) \
  __builtin_overload(1, x, ccoshl, ccosh, ccoshf, coshl, cosh, coshf)
#define sinh(x) \
  __builtin_overload(1, x, csinhl, csinh, csinhf, sinhl, sinh, sinhf)
#define tanh(x) \
  __builtin_overload(1, x, ctanhl, ctanh, ctanhf, tanhl, tanh, tanhf)
#define exp(x) \
  __builtin_overload(1, x, cexpl, cexp, cexpf, expl, exp, expf)
#define log(x) \
  __builtin_overload(1, x, clogl, clog, clogf, logl, log, logf)
#define sqrt(x) \
  __builtin_overload(1, x, csqrtl, csqrt, csqrtf, sqrtl, sqrt, sqrtf)
#define fabs(x) \
  __builtin_overload(1, x, cabsl, cabs, cabsf, fabsl, fabs, fabsf)
// FIXME: POW -> binary operation.

/* C99 7.22p5, functions in just math.h that have no complex counterpart. */

// FIXME: atan2 -> binary operation.
#define cbrt(x)  __builtin_overload(1, x, cbrtl, cbrt, cbrtf)
#define ceil(x)  __builtin_overload(1, x, ceill, ceil, ceilf)
// FIXME: copysign -> binary operation.
#define erf(x)   __builtin_overload(1, x, erfl, erf, erff)
#define erfc(x)  __builtin_overload(1, x, erfcl, erfc, erfcf)
#define exp2(x)  __builtin_overload(1, x, expl, exp2, exp2f)
#define expm1(x) __builtin_overload(1, x, expm1l, expm1, expm1f)
// FIXME: fdim -> binary operation.
#define floor(x) __builtin_overload(1, x, floorl, floor, floorf)
// FIXME: fma -> trinary operation.
// FIXME: fmax -> binary operation.
// FIXME: fmin -> binary operation.
// FIXME: fmax -> binary operation.
// FIXME: fmod -> binary operation.
// FIXME: frexp -> unary + pointer operation.
// FIXME: hypot -> binary operation.
#define ilogb(x) __builtin_overload(1, x, ilogbl, ilogb, ilogbf)
// FIXME: ldexp -> fp+int.
#define lgamma(x) __builtin_overload(1, x, lgammal, lgamma, lgammaf)
#define llrint(x) __builtin_overload(1, x, llrintl, llrint, llrintf)
#define llround(x) __builtin_overload(1, x, llroundl, llround, llroundf)
#define log10(x)   __builtin_overload(1, x, log10l, log10, log10f)
#define log1p(x)  __builtin_overload(1, x, log1pl, log1p, log1pf)
#define log2(x)   __builtin_overload(1, x, log2l, log2, log2f)
#define logb(x)   __builtin_overload(1, x, logbl, logb, logbf)
#define lrint(x)  __builtin_overload(1, x, lrintl, lrint, lrintf)
#define lround(x) __builtin_overload(1, x, lroundl, lround, lroundf)
#define nearbyint(x)__builtin_overload(1, x, nearbyintl, nearbyint, nearbyintf)
// FIXME: nextafter -> binary operation.
// FIXME: nexttoward -> binary operation?  [second arg is always long double]
// FIXME: remainder -> binary operation.
// FIXME: remquo -> fp+fp+ptr
#define rint(x) __builtin_overload(1, x, rintl, rint, rintf)
#define round(x) __builtin_overload(1, x, roundl, round, roundf)
// FIXME: scalbn -> fp+int
// FIXME: scalbln -> fp+int
#define tgamma(x) __builtin_overload(1, x, tgammal, tgamma, tgammaf)
#define trunc(x) __builtin_overload(1, x, truncl, trunc, truncf)


// FIXME: carg, cimag, conj, cproj, creal

#endif /* __cplusplus */
#endif /* __TGMATH_H */
