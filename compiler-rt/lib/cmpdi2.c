/* ===-- cmpdi2.c - Implement __cmpdi2 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 * 
 * This file implements __cmpdi2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

/* Returns:  if (a <  b) returns 0
*           if (a == b) returns 1
*           if (a >  b) returns 2
*/

si_int
__cmpdi2(di_int a, di_int b)
{
    dwords x;
    x.all = a;
    dwords y;
    y.all = b;
    if (x.high < y.high)
        return 0;
    if (x.high > y.high)
        return 2;
    if (x.low < y.low)
        return 0;
    if (x.low > y.low)
        return 2;
    return 1;
}
