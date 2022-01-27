//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <type_traits>

// template <class T> struct is_aggregate;
// template <class T> constexpr bool is_aggregate_v = is_aggregate<T>::value;

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_true()
{
    static_assert( std::is_aggregate<T>::value, "");
    static_assert( std::is_aggregate<const T>::value, "");
    static_assert( std::is_aggregate<volatile T>::value, "");
    static_assert( std::is_aggregate<const volatile T>::value, "");
    static_assert( std::is_aggregate_v<T>, "");
    static_assert( std::is_aggregate_v<const T>, "");
    static_assert( std::is_aggregate_v<volatile T>, "");
    static_assert( std::is_aggregate_v<const volatile T>, "");
}

template <class T>
void test_false()
{
    static_assert(!std::is_aggregate<T>::value, "");
    static_assert(!std::is_aggregate<const T>::value, "");
    static_assert(!std::is_aggregate<volatile T>::value, "");
    static_assert(!std::is_aggregate<const volatile T>::value, "");
    static_assert(!std::is_aggregate_v<T>, "");
    static_assert(!std::is_aggregate_v<const T>, "");
    static_assert(!std::is_aggregate_v<volatile T>, "");
    static_assert(!std::is_aggregate_v<const volatile T>, "");
}

struct Aggregate {};
struct HasCons { HasCons(int); };
struct HasPriv {
  void PreventUnusedPrivateMemberWarning();
private:
  int x;
};
struct Union { int x; void* y; };


int main(int, char**)
{
  {
    test_false<void>();
    test_false<int>();
    test_false<void*>();
    test_false<void()>();
    test_false<void() const>();
    test_false<void(Aggregate::*)(int) const>();
    test_false<Aggregate&>();
    test_false<HasCons>();
    test_false<HasPriv>();
  }
  {
    test_true<Aggregate>();
    test_true<Aggregate[]>();
    test_true<Aggregate[42][101]>();
    test_true<Union>();
  }

  return 0;
}
