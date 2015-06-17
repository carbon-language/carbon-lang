//===-- c.c -----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
int
d_function ()
{ // Find this line number within d_dunction().
#ifdef HIDDEN
    return 12345;
#else
    return 700;
#endif
}
