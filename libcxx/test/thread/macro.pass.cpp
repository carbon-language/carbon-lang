//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// #define __STDCPP_THREADS__ __cplusplus

#include <thread>

int main()
{
#ifndef __STDCPP_THREADS__
#error __STDCPP_THREADS__ is not defined
#endif
}
