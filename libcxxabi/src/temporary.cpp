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
void f1()
{
    abort_message("__cxa_new_handler shouldn't be called");
}

static
void f2()
{
    abort_message("__cxa_terminate_handler shouldn't be called");
}

static
void f3()
{
    abort_message("__cxa_unexpected_handler shouldn't be called");
}

void (*__cxa_new_handler)() = f1;
void (*__cxa_terminate_handler)() = f2;
void (*__cxa_unexpected_handler)() = f3;

}

#pragma GCC visibility pop
