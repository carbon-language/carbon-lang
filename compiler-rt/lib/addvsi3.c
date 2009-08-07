/* ===-- addvsi3.c - Implement __addvsi3 -----------------------------------===
 *
 *                    The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __addvsi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include <stdlib.h>

/* Returns: a + b */

/* Effects: aborts if a + b overflows */

si_int
__addvsi3(si_int a, si_int b)
{
    si_int s = a + b;
    if (b >= 0)
    {
        if (s < a)
            abort();
    }
    else
    {
        if (s >= a)
            abort();
    }
    return s;
}
