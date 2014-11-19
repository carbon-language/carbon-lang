//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <stdarg.h>

#include <stdarg.h>

#ifndef va_arg
#error va_arg not defined
#endif

#if __cplusplus >= 201103L
#  ifndef va_copy
#    error va_copy is not defined when c++ >= 11
#  endif
#else
#  ifdef va_copy
#    error va_copy is unexpectedly defined when c++ < 11
#  endif
#endif

#ifndef va_end
#error va_end not defined
#endif

#ifndef va_start
#error va_start not defined
#endif

int main()
{
    va_list va;
}
