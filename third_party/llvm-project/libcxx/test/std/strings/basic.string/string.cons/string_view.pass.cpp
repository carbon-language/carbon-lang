//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// explicit basic_string(basic_string_view<CharT, traits> sv, const Allocator& a = Allocator()); // constexpr since C++20

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string_view>
#include <string>
#include <type_traits>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

static_assert(!std::is_convertible<std::string_view, std::string const&>::value, "");
static_assert(!std::is_convertible<std::string_view, std::string>::value, "");

template <class charT>
TEST_CONSTEXPR_CXX20 void
test(std::basic_string_view<charT> sv)
{
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
  {
    S s2(sv);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
  }
  {
    S s2;
    s2 = sv;
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
  }
}

template <class charT, class A>
TEST_CONSTEXPR_CXX20 void
test(std::basic_string_view<charT> sv, const A& a)
{
    typedef std::basic_string<charT, std::char_traits<charT>, A> S;
    typedef typename S::traits_type T;
  {
    S s2(sv, a);
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
  }
  {
    S s2(a);
    s2 = sv;
    LIBCPP_ASSERT(s2.__invariants());
    assert(s2.size() == sv.size());
    assert(T::compare(s2.data(), sv.data(), sv.size()) == 0);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef test_allocator<char> A;
    typedef std::basic_string_view<char, std::char_traits<char> > SV;

    test(SV(""));
    test(SV(""), A(2));

    test(SV("1"));
    test(SV("1") ,A(2));

    test(SV("1234567980"));
    test(SV("1234567980"), A(2));

    test(SV("123456798012345679801234567980123456798012345679801234567980"));
    test(SV("123456798012345679801234567980123456798012345679801234567980"), A(2));
  }
#if TEST_STD_VER >= 11
  {
    typedef min_allocator<char> A;
    typedef std::basic_string_view<char, std::char_traits<char> > SV;

    test(SV(""));
    test(SV(""), A());

    test(SV("1"));
    test(SV("1") ,A());

    test(SV("1234567980"));
    test(SV("1234567980"), A());

    test(SV("123456798012345679801234567980123456798012345679801234567980"));
    test(SV("123456798012345679801234567980123456798012345679801234567980"), A());
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
