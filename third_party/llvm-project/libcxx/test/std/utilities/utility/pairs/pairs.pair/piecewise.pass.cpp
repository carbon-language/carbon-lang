//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// template <class... Args1, class... Args2>
//     pair(piecewise_construct_t, tuple<Args1...> first_args,
//                                 tuple<Args2...> second_args);

#include <cassert>
#include <tuple>
#include <utility>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::pair<int, int*> P1;
    typedef std::pair<int*, int> P2;
    typedef std::pair<P1, P2> P3;
    P3 p3(std::piecewise_construct, std::tuple<int, int*>(3, nullptr),
          std::tuple<int*, int>(nullptr, 4));
    assert(p3.first == P1(3, nullptr));
    assert(p3.second == P2(nullptr, 4));
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
