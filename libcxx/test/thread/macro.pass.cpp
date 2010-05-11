//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// #define __STDCPP_THREADS __cplusplus

#include <thread>

int main()
{
#ifndef __STDCPP_THREADS
#error __STDCPP_THREADS is not defined
#endif
}
