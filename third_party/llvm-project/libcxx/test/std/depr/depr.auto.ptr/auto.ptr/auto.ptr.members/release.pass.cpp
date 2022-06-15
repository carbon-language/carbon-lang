//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// X* release() throw();

// REQUIRES: c++03 || c++11 || c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "../A.h"

void
test()
{
    {
    A* p = new A(1);
    std::auto_ptr<A> ap(p);
    A* p2 = ap.release();
    assert(p2 == p);
    assert(ap.get() == 0);
    delete p;
    }
    assert(A::count == 0);
}

int main(int, char**)
{
    test();

  return 0;
}
