/*===-- divsc3.c - Implement __divsc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divsc3 for the compiler_rt library.
 *
 *===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include "int_math.h"
#include <math.h>

/* Returns: the quotient of (a + ib) / (c + id) */

float _Complex
__divsc3(float __a, float __b, float __c, float __d)
{
    int __ilogbw = 0;
    float __logbw = logbf(fmaxf(fabsf(__c), fabsf(__d)));
    if (crt_isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = scalbnf(__c, -__ilogbw);
        __d = scalbnf(__d, -__ilogbw);
    }
    float __denom = __c * __c + __d * __d;
    float _Complex z;
    __real__ z = scalbnf((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = scalbnf((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (crt_isnan(__real__ z) && crt_isnan(__imag__ z))
    {
        if ((__denom == 0) && (!crt_isnan(__a) || !crt_isnan(__b)))
        {
            __real__ z = copysignf(INFINITY, __c) * __a;
            __imag__ z = copysignf(INFINITY, __c) * __b;
        }
        else if ((crt_isinf(__a) || crt_isinf(__b)) &&
                 crt_isfinite(__c) && crt_isfinite(__d))
        {
            __a = copysignf(crt_isinf(__a) ? 1 : 0, __a);
            __b = copysignf(crt_isinf(__b) ? 1 : 0, __b);
            __real__ z = INFINITY * (__a * __c + __b * __d);
            __imag__ z = INFINITY * (__b * __c - __a * __d);
        }
        else if (crt_isinf(__logbw) && __logbw > 0 &&
                 crt_isfinite(__a) && crt_isfinite(__b))
        {
            __c = copysignf(crt_isinf(__c) ? 1 : 0, __c);
            __d = copysignf(crt_isinf(__d) ? 1 : 0, __d);
            __real__ z = 0 * (__a * __c + __b * __d);
            __imag__ z = 0 * (__b * __c - __a * __d);
        }
    }
    return z;
}
