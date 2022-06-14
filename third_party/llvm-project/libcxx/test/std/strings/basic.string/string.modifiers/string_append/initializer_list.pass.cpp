//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string& append(initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool test() {
  {
    std::string s("123");
    s.append({'a', 'b', 'c'});
    assert(s == "123abc");
  }
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s("123");
    s.append({'a', 'b', 'c'});
    assert(s == "123abc");
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
