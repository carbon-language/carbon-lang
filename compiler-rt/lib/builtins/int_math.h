/* ===-- int_math.h - internal math inlines ---------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===-----------------------------------------------------------------------===
 *
 * This file is not part of the interface of this library.
 *
 * This file defines substitutes for the libm functions used in some of the
 * compiler-rt implementations, defined in such a way that there is not a direct
 * dependency on libm or math.h. Instead, we use the compiler builtin versions
 * where available. This reduces our dependencies on the system SDK by foisting
 * the responsibility onto the compiler.
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef INT_MATH_H
#define INT_MATH_H

#ifndef __has_builtin
#  define  __has_builtin(x) 0
#endif

#define CRT_INFINITY __builtin_huge_valf()

#define crt_isinf(x) __builtin_isinf((x))
#define crt_isnan(x) __builtin_isnan((x))

/* Define crt_isfinite in terms of the builtin if available, otherwise provide
 * an alternate version in terms of our other functions. This supports some
 * versions of GCC which didn't have __builtin_isfinite.
 */
#if __has_builtin(__builtin_isfinite)
#  define crt_isfinite(x) __builtin_isfinite((x))
#else
#  define crt_isfinite(x) \
  __extension__(({ \
      __typeof((x)) x_ = (x); \
      !crt_isinf(x_) && !crt_isnan(x_); \
    }))
#endif

#define crt_copysign(x, y) __builtin_copysign((x), (y))
#define crt_copysignf(x, y) __builtin_copysignf((x), (y))
#define crt_copysignl(x, y) __builtin_copysignl((x), (y))

#define crt_fabs(x) __builtin_fabs((x))
#define crt_fabsf(x) __builtin_fabsf((x))
#define crt_fabsl(x) __builtin_fabsl((x))

#define crt_fmax(x, y) __builtin_fmax((x), (y))
#define crt_fmaxf(x, y) __builtin_fmaxf((x), (y))
#define crt_fmaxl(x, y) __builtin_fmaxl((x), (y))

#define crt_logb(x) __builtin_logb((x))
#define crt_logbf(x) __builtin_logbf((x))
#define crt_logbl(x) __builtin_logbl((x))

#define crt_scalbn(x, y) __builtin_scalbn((x), (y))
#define crt_scalbnf(x, y) __builtin_scalbnf((x), (y))
#define crt_scalbnl(x, y) __builtin_scalbnl((x), (y))

#endif /* INT_MATH_H */
