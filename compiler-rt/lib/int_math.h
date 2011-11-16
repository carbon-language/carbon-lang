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

#define crt_isfinite(x) __builtin_isfinite((x))
#define crt_isinf(x) __builtin_isinf((x))
#define crt_isnan(x) __builtin_isnan((x))

#endif /* INT_MATH_H */
