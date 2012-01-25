//===---------------------------- temporary.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma GCC visibility push(default)

extern "C"
{

void (*__cxa_new_handler)() = 0;
void (*__cxa_terminate_handler)() = 0;
void (*__cxa_unexpected_handler)() = 0;

}

#pragma GCC visibility pop
