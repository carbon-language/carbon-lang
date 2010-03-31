/* ===-- negvdi2.c - Implement __negvdi2 -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __negvdi2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include <stdlib.h>

/* Returns: -a */

/* Effects: aborts if -a overflows */

di_int
__negvdi2(di_int a)
{
    const di_int MIN = (di_int)1 << ((int)(sizeof(di_int) * CHAR_BIT)-1);
    if (a == MIN)
        compilerrt_abort();
    return -a;
}
