// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_variant_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// <variant>

// template <class ...Types> class variant;

// template <size_t I, class U, class ...Args>
//   variant_alternative_t<I, variant<Types...>>& emplace(initializer_list<U> il,Args&&... args);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"

struct InitList {
  std::size_t size;
  constexpr InitList(std::initializer_list<int> il) : size(il.size()) {}
};

struct InitListArg {
  std::size_t size;
  int value;
  constexpr InitListArg(std::initializer_list<int> il, int v)
      : size(il.size()), value(v) {}
};

template <class Var, size_t I, class... Args>
constexpr auto test_emplace_exists_imp(int) -> decltype(
    std::declval<Var>().template emplace<I>(std::declval<Args>()...), true) {
  return true;
}

template <class, size_t, class...>
constexpr auto test_emplace_exists_imp(long) -> bool {
  return false;
}

template <class Var, size_t I, class... Args> constexpr bool emplace_exists() {
  return test_emplace_exists_imp<Var, I, Args...>(0);
}

void test_emplace_sfinae() {
  using V =
      std::variant<int, TestTypes::NoCtors, InitList, InitListArg, long, long>;
  using IL = std::initializer_list<int>;
  static_assert(!emplace_exists<V, 1, IL>(), "no such constructor");
  static_assert(emplace_exists<V, 2, IL>(), "");
  static_assert(!emplace_exists<V, 2, int>(), "args don't match");
  static_assert(!emplace_exists<V, 2, IL, int>(), "too many args");
  static_assert(emplace_exists<V, 3, IL, int>(), "");
  static_assert(!emplace_exists<V, 3, int>(), "args don't match");
  static_assert(!emplace_exists<V, 3, IL>(), "too few args");
  static_assert(!emplace_exists<V, 3, IL, int, int>(), "too many args");
}

void test_basic() {
  using V = std::variant<int, InitList, InitListArg, TestTypes::NoCtors>;
  V v;
  auto& ref1 = v.emplace<1>({1, 2, 3});
  static_assert(std::is_same_v<InitList&, decltype(ref1)>, "");
  assert(std::get<1>(v).size == 3);
  assert(&ref1 == &std::get<1>(v));
  auto& ref2 = v.emplace<2>({1, 2, 3, 4}, 42);
  static_assert(std::is_same_v<InitListArg&, decltype(ref2)>, "");
  assert(std::get<2>(v).size == 4);
  assert(std::get<2>(v).value == 42);
  assert(&ref2 == &std::get<2>(v));
  auto& ref3 = v.emplace<1>({1});
  static_assert(std::is_same_v<InitList&, decltype(ref3)>, "");
  assert(std::get<1>(v).size == 1);
  assert(&ref3 == &std::get<1>(v));
}

int main(int, char**) {
  test_basic();
  test_emplace_sfinae();

  return 0;
}
