/* ===-- udivdi3.c - Implement __udivdi3 -----------------------------------===
 *
 *                    The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __udivdi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

du_int __udivmoddi4(du_int a, du_int b, du_int* rem);

/* Returns: a / b */

du_int
__udivdi3(du_int a, du_int b)
{
    return __udivmoddi4(a, b, 0);
}
