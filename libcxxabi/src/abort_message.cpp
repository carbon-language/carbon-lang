//===-------------------------- abort_message.c ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "abort_message.h"

__attribute__((visibility("hidden")))
void abort_message(const char* format, ...)
{
    // write message to stderr
#if __APPLE__
    fprintf(stderr, "libc++abi.dylib: ");
#endif
    va_list list;
    va_start(list, format);
    vfprintf(stderr, format, list);
    va_end(list);
    fprintf(stderr, "\n");
    abort();
}

