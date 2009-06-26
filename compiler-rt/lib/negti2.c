//===-- negti2.c - Implement __negti2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements __negti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"

// Returns: -a

ti_int
__negti2(ti_int a)
{
    return ~a + 1;
}

#endif
