//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable S, ClassType T, CopyConstructible A>
//   const_mem_fun1_t<S,T,A>
//   mem_fun(S (T::*f)(A) const);
// Removed in c++17
// UNSUPPORTED: c++03, c++11, c++14

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
    char a1() {return 5;}
    short a2(int i) {return short(i+1);}
    int a3() const {return 1;}
    double a4(unsigned i) const {return i-1;}
};

int main(int, char**)
{
    const A a = A();
    assert(std::mem_fun(&A::a4)(&a, 6) == 5);

  return 0;
}
