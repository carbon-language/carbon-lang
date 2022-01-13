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

// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>>&
//   get(variant<Types...>& v);
// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>>&&
//   get(variant<Types...>&& v);
// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>> const& get(const
//   variant<Types...>& v);
// template <size_t I, class... Types>
//  constexpr variant_alternative_t<I, variant<Types...>> const&& get(const
//  variant<Types...>&& v);

#include "test_macros.h"
#include "test_workarounds.h"
#include "variant_test_helpers.h"
#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

void test_const_lvalue_get() {
  {
    using V = std::variant<int, const long>;
    constexpr V v(42);
    ASSERT_NOT_NOEXCEPT(std::get<0>(v));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), const int &);
    static_assert(std::get<0>(v) == 42, "");
  }
  {
    using V = std::variant<int, const long>;
    const V v(42);
    ASSERT_NOT_NOEXCEPT(std::get<0>(v));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), const int &);
    assert(std::get<0>(v) == 42);
  }
  {
    using V = std::variant<int, const long>;
    constexpr V v(42l);
    ASSERT_NOT_NOEXCEPT(std::get<1>(v));
    ASSERT_SAME_TYPE(decltype(std::get<1>(v)), const long &);
    static_assert(std::get<1>(v) == 42, "");
  }
  {
    using V = std::variant<int, const long>;
    const V v(42l);
    ASSERT_NOT_NOEXCEPT(std::get<1>(v));
    ASSERT_SAME_TYPE(decltype(std::get<1>(v)), const long &);
    assert(std::get<1>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int &>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), int &);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<int &&>;
    int x = 42;
    const V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), int &);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<const int &&>;
    int x = 42;
    const V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), const int &);
    assert(&std::get<0>(v) == &x);
  }
#endif
}

void test_lvalue_get() {
  {
    using V = std::variant<int, const long>;
    V v(42);
    ASSERT_NOT_NOEXCEPT(std::get<0>(v));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), int &);
    assert(std::get<0>(v) == 42);
  }
  {
    using V = std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(std::get<1>(v)), const long &);
    assert(std::get<1>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int &>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), int &);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<const int &>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), const int &);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<int &&>;
    int x = 42;
    V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), int &);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<const int &&>;
    int x = 42;
    V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(v)), const int &);
    assert(&std::get<0>(v) == &x);
  }
#endif
}

void test_rvalue_get() {
  {
    using V = std::variant<int, const long>;
    V v(42);
    ASSERT_NOT_NOEXCEPT(std::get<0>(std::move(v)));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), int &&);
    assert(std::get<0>(std::move(v)) == 42);
  }
  {
    using V = std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(std::get<1>(std::move(v))), const long &&);
    assert(std::get<1>(std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int &>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), int &);
    assert(&std::get<0>(std::move(v)) == &x);
  }
  {
    using V = std::variant<const int &>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), const int &);
    assert(&std::get<0>(std::move(v)) == &x);
  }
  {
    using V = std::variant<int &&>;
    int x = 42;
    V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), int &&);
    int &&xref = std::get<0>(std::move(v));
    assert(&xref == &x);
  }
  {
    using V = std::variant<const int &&>;
    int x = 42;
    V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), const int &&);
    const int &&xref = std::get<0>(std::move(v));
    assert(&xref == &x);
  }
#endif
}

void test_const_rvalue_get() {
  {
    using V = std::variant<int, const long>;
    const V v(42);
    ASSERT_NOT_NOEXCEPT(std::get<0>(std::move(v)));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), const int &&);
    assert(std::get<0>(std::move(v)) == 42);
  }
  {
    using V = std::variant<int, const long>;
    const V v(42l);
    ASSERT_SAME_TYPE(decltype(std::get<1>(std::move(v))), const long &&);
    assert(std::get<1>(std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int &>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), int &);
    assert(&std::get<0>(std::move(v)) == &x);
  }
  {
    using V = std::variant<const int &>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), const int &);
    assert(&std::get<0>(std::move(v)) == &x);
  }
  {
    using V = std::variant<int &&>;
    int x = 42;
    const V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), int &&);
    int &&xref = std::get<0>(std::move(v));
    assert(&xref == &x);
  }
  {
    using V = std::variant<const int &&>;
    int x = 42;
    const V v(std::move(x));
    ASSERT_SAME_TYPE(decltype(std::get<0>(std::move(v))), const int &&);
    const int &&xref = std::get<0>(std::move(v));
    assert(&xref == &x);
  }
#endif
}

template <std::size_t I> using Idx = std::integral_constant<size_t, I>;

void test_throws_for_all_value_categories() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<int, long>;
  V v0(42);
  const V &cv0 = v0;
  assert(v0.index() == 0);
  V v1(42l);
  const V &cv1 = v1;
  assert(v1.index() == 1);
  std::integral_constant<size_t, 0> zero;
  std::integral_constant<size_t, 1> one;
  auto test = [](auto idx, auto &&v) {
    using Idx = decltype(idx);
    try {
      TEST_IGNORE_NODISCARD std::get<Idx::value>(std::forward<decltype(v)>(v));
    } catch (const std::bad_variant_access &) {
      return true;
    } catch (...) { /* ... */
    }
    return false;
  };
  { // lvalue test cases
    assert(test(one, v0));
    assert(test(zero, v1));
  }
  { // const lvalue test cases
    assert(test(one, cv0));
    assert(test(zero, cv1));
  }
  { // rvalue test cases
    assert(test(one, std::move(v0)));
    assert(test(zero, std::move(v1)));
  }
  { // const rvalue test cases
    assert(test(one, std::move(cv0)));
    assert(test(zero, std::move(cv1)));
  }
#endif
}

int main(int, char**) {
  test_const_lvalue_get();
  test_lvalue_get();
  test_rvalue_get();
  test_const_rvalue_get();
  test_throws_for_all_value_categories();

  return 0;
}
