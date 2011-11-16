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
#include <math.h>

/* Returns: the quotient of (a + ib) / (c + id) */

double _Complex
__divdc3(double __a, double __b, double __c, double __d)
{
    int __ilogbw = 0;
    double __logbw = logb(fmax(fabs(__c), fabs(__d)));
    if (crt_isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = scalbn(__c, -__ilogbw);
        __d = scalbn(__d, -__ilogbw);
    }
    double __denom = __c * __c + __d * __d;
    double _Complex z;
    __real__ z = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (crt_isnan(__real__ z) && crt_isnan(__imag__ z))
    {
        if ((__denom == 0.0) && (!crt_isnan(__a) || !crt_isnan(__b)))
        {
            __real__ z = copysign(INFINITY, __c) * __a;
            __imag__ z = copysign(INFINITY, __c) * __b;
        }
        else if ((crt_isinf(__a) || crt_isinf(__b)) &&
                 crt_isfinite(__c) && crt_isfinite(__d))
        {
            __a = copysign(crt_isinf(__a) ? 1.0 : 0.0, __a);
            __b = copysign(crt_isinf(__b) ? 1.0 : 0.0, __b);
            __real__ z = INFINITY * (__a * __c + __b * __d);
            __imag__ z = INFINITY * (__b * __c - __a * __d);
        }
        else if (crt_isinf(__logbw) && __logbw > 0.0 &&
                 crt_isfinite(__a) && crt_isfinite(__b))
        {
            __c = copysign(crt_isinf(__c) ? 1.0 : 0.0, __c);
            __d = copysign(crt_isinf(__d) ? 1.0 : 0.0, __d);
            __real__ z = 0.0 * (__a * __c + __b * __d);
            __imag__ z = 0.0 * (__b * __c - __a * __d);
        }
    }
    return z;
}
