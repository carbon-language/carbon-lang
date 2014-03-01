/* ===-- divdc3.c - Implement __divdc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divdc3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include "int_math.h"

/* Returns: the quotient of (a + ib) / (c + id) */

COMPILER_RT_ABI double _Complex
__divdc3(double __a, double __b, double __c, double __d)
{
    int __ilogbw = 0;
    double __logbw = crt_logb(crt_fmax(crt_fabs(__c), crt_fabs(__d)));
    if (crt_isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = crt_scalbn(__c, -__ilogbw);
        __d = crt_scalbn(__d, -__ilogbw);
    }
    double __denom = __c * __c + __d * __d;
    double _Complex z;
    __real__ z = crt_scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = crt_scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (crt_isnan(__real__ z) && crt_isnan(__imag__ z))
    {
        if ((__denom == 0.0) && (!crt_isnan(__a) || !crt_isnan(__b)))
        {
            __real__ z = crt_copysign(CRT_INFINITY, __c) * __a;
            __imag__ z = crt_copysign(CRT_INFINITY, __c) * __b;
        }
        else if ((crt_isinf(__a) || crt_isinf(__b)) &&
                 crt_isfinite(__c) && crt_isfinite(__d))
        {
            __a = crt_copysign(crt_isinf(__a) ? 1.0 : 0.0, __a);
            __b = crt_copysign(crt_isinf(__b) ? 1.0 : 0.0, __b);
            __real__ z = CRT_INFINITY * (__a * __c + __b * __d);
            __imag__ z = CRT_INFINITY * (__b * __c - __a * __d);
        }
        else if (crt_isinf(__logbw) && __logbw > 0.0 &&
                 crt_isfinite(__a) && crt_isfinite(__b))
        {
            __c = crt_copysign(crt_isinf(__c) ? 1.0 : 0.0, __c);
            __d = crt_copysign(crt_isinf(__d) ? 1.0 : 0.0, __d);
            __real__ z = 0.0 * (__a * __c + __b * __d);
            __imag__ z = 0.0 * (__b * __c - __a * __d);
        }
    }
    return z;
}
