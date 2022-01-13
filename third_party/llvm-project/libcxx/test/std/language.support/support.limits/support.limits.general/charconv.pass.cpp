//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <charconv> feature macros

/*  Constant                                    Value
    __cpp_lib_to_chars                          201611L

*/

#include <charconv>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
//  ensure that the macros that are supposed to be defined in <utility> are defined.

/*
#if !defined(__cpp_lib_fooby)
# error "__cpp_lib_fooby is not defined"
#elif __cpp_lib_fooby < 201606L
# error "__cpp_lib_fooby has an invalid value"
#endif
*/

  return 0;
}
