//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//


#include <span>
#include <array>
#include <cassert>
#include <string>

#include "test_macros.h"

void checkCV()
{
    std::array<int, 3> arr  = {1,2,3};
//  STL says these are not cromulent
//  std::array<const int,3> carr = {4,5,6};
//  std::array<volatile int, 3> varr = {7,8,9};
//  std::array<const volatile int, 3> cvarr = {1,3,5};

//  Types the same (dynamic sized)
    {
    std::span<               int> s1{  arr};    // a span<               int> pointing at int.
    }

//  Types the same (static sized)
    {
    std::span<               int,3> s1{  arr};  // a span<               int> pointing at int.
    }


//  types different (dynamic sized)
    {
    std::span<const          int> s1{ arr};     // a span<const          int> pointing at int.
    std::span<      volatile int> s2{ arr};     // a span<      volatile int> pointing at int.
    std::span<      volatile int> s3{ arr};     // a span<      volatile int> pointing at const int.
    std::span<const volatile int> s4{ arr};     // a span<const volatile int> pointing at int.
    }

//  types different (static sized)
    {
    std::span<const          int,3> s1{ arr};   // a span<const          int> pointing at int.
    std::span<      volatile int,3> s2{ arr};   // a span<      volatile int> pointing at int.
    std::span<      volatile int,3> s3{ arr};   // a span<      volatile int> pointing at const int.
    std::span<const volatile int,3> s4{ arr};   // a span<const volatile int> pointing at int.
    }
}

template <typename T, typename U>
constexpr bool testConstructorArray() {
  std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(std::span<T>{val});
  ASSERT_NOEXCEPT(std::span<T, 2>{val});
  std::span<T> s1{val};
  std::span<T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T, typename U>
constexpr bool testConstructorConstArray() {
  const std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(std::span<const T>{val});
  ASSERT_NOEXCEPT(std::span<const T, 2>{val});
  std::span<const T> s1{val};
  std::span<const T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T>
constexpr bool testConstructors() {
  static_assert(testConstructorArray<T, T>(), "");
  static_assert(testConstructorArray<const T, const T>(), "");
  static_assert(testConstructorArray<const T, T>(), "");
  static_assert(testConstructorConstArray<T, T>(), "");
  static_assert(testConstructorConstArray<const T, const T>(), "");
  static_assert(testConstructorConstArray<const T, T>(), "");

  return testConstructorArray<T, T>() &&
         testConstructorArray<const T, const T>() &&
         testConstructorArray<const T, T>() &&
         testConstructorConstArray<T, T>() &&
         testConstructorConstArray<const T, const T>() &&
         testConstructorConstArray<const T, T>();
}

struct A{};

int main(int, char**)
{
    assert(testConstructors<int>());
    assert(testConstructors<long>());
    assert(testConstructors<double>());
    assert(testConstructors<A>());

    assert(testConstructors<int*>());
    assert(testConstructors<const int*>());

    checkCV();

    return 0;
}
