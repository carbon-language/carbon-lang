//===------------------------- cxa_unexpected.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <exception>

#pragma GCC visibility push(default)

extern "C"
{

__attribute__((noreturn))
void
__cxa_call_unexpected (void*)
{
    // TODO:  Completely unfinished!
    std::terminate();
}

}

#pragma GCC visibility pop
