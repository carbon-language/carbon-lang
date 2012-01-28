//===---------------------------- temporary.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "abort_message.h"

#pragma GCC visibility push(default)

extern "C"
{

static
void f()
{
    abort_message("this shouldn't be called");
}

void (*__cxa_new_handler)() = f;
void (*__cxa_terminate_handler)() = f;
void (*__cxa_unexpected_handler)() = f;

}

#pragma GCC visibility pop
