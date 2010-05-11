//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <cstdarg>

#include <cstdarg>

#ifndef va_arg
#error va_arg not defined
#endif

#ifndef va_copy
#error va_copy not defined
#endif

#ifndef va_end
#error va_end not defined
#endif

#ifndef va_start
#error va_start not defined
#endif

int main()
{
    std::va_list va;
}
