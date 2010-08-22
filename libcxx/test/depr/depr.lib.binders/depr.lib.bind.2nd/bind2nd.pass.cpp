//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class Fn, class T>
//   binder2nd<Fn>
//   bind2nd(const Fn& op, const T& x);

#include <functional>
#include <cassert>

#include "../test_func.h"

int main()
{
    assert(std::bind2nd(test_func(1), 5)(10) == 5.);
}
