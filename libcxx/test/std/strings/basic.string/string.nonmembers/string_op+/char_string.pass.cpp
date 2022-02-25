//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>
//   operator+(charT lhs, const basic_string<charT,traits,Allocator>& rhs);

// template<class charT, class traits, class Allocator>
//   basic_string<charT,traits,Allocator>&&
//   operator+(charT lhs, basic_string<charT,traits,Allocator>&& rhs);

#include <string>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test0(typename S::value_type lhs, const S& rhs, const S& x) {
  assert(lhs + rhs == x);
}

#if TEST_STD_VER >= 11
template <class S>
TEST_CONSTEXPR_CXX20 void test1(typename S::value_type lhs, S&& rhs, const S& x) {
  assert(lhs + std::move(rhs) == x);
}
#endif

bool test() {
  {
    typedef std::string S;
    test0('a', S(""), S("a"));
    test0('a', S("12345"), S("a12345"));
    test0('a', S("1234567890"), S("a1234567890"));
    test0('a', S("12345678901234567890"), S("a12345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::string S;
    test1('a', S(""), S("a"));
    test1('a', S("12345"), S("a12345"));
    test1('a', S("1234567890"), S("a1234567890"));
    test1('a', S("12345678901234567890"), S("a12345678901234567890"));
  }
  {
    typedef std::basic_string<char, std::char_traits<char>,
                              min_allocator<char> >
        S;
    test0('a', S(""), S("a"));
    test0('a', S("12345"), S("a12345"));
    test0('a', S("1234567890"), S("a1234567890"));
    test0('a', S("12345678901234567890"), S("a12345678901234567890"));

    test1('a', S(""), S("a"));
    test1('a', S("12345"), S("a12345"));
    test1('a', S("1234567890"), S("a1234567890"));
    test1('a', S("12345678901234567890"), S("a12345678901234567890"));
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
