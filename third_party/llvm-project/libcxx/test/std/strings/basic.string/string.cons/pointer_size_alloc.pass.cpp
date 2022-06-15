//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const charT* s, size_type n, const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class charT>
TEST_CONSTEXPR_CXX20 void
test(const charT* s, unsigned n)
{
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    S s2(s, n);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == n);
    assert(T::compare(s2.data(), s, n) == 0);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
}

template <class charT, class A>
TEST_CONSTEXPR_CXX20 void
test(const charT* s, unsigned n, const A& a)
{
    typedef std::basic_string<charT, std::char_traits<charT>, A> S;
    typedef typename S::traits_type T;
    S s2(s, n, a);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == n);
    assert(T::compare(s2.data(), s, n) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef test_allocator<char> A;

    test("", 0);
    test("", 0, A(2));

    test("1", 1);
    test("1", 1, A(2));

    test("1234567980", 10);
    test("1234567980", 10, A(2));

    test("123456798012345679801234567980123456798012345679801234567980", 60);
    test("123456798012345679801234567980123456798012345679801234567980", 60, A(2));
  }
#if TEST_STD_VER >= 11
  {
    typedef min_allocator<char> A;

    test("", 0);
    test("", 0, A());

    test("1", 1);
    test("1", 1, A());

    test("1234567980", 10);
    test("1234567980", 10, A());

    test("123456798012345679801234567980123456798012345679801234567980", 60);
    test("123456798012345679801234567980123456798012345679801234567980", 60, A());
  }
#endif

#if TEST_STD_VER > 3
  {   // LWG 2946
    std::string s({"abc", 1});
    assert(s.size() == 1);
    assert(s == "a");
  }
#endif

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
