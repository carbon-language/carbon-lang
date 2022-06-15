//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// class strong_ordering


#include <compare>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

const volatile void* volatile sink;

void test_static_members() {
  DoNotOptimize(&std::strong_ordering::less);
  DoNotOptimize(&std::strong_ordering::equal);
  DoNotOptimize(&std::strong_ordering::equivalent);
  DoNotOptimize(&std::strong_ordering::greater);
}

void test_signatures() {
  auto& Eq = std::strong_ordering::equivalent;

  ASSERT_NOEXCEPT(Eq == 0);
  ASSERT_NOEXCEPT(0 == Eq);
  ASSERT_NOEXCEPT(Eq != 0);
  ASSERT_NOEXCEPT(0 != Eq);
  ASSERT_NOEXCEPT(0 < Eq);
  ASSERT_NOEXCEPT(Eq < 0);
  ASSERT_NOEXCEPT(0 <= Eq);
  ASSERT_NOEXCEPT(Eq <= 0);
  ASSERT_NOEXCEPT(0 > Eq);
  ASSERT_NOEXCEPT(Eq > 0);
  ASSERT_NOEXCEPT(0 >= Eq);
  ASSERT_NOEXCEPT(Eq >= 0);
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  ASSERT_NOEXCEPT(0 <=> Eq);
  ASSERT_NOEXCEPT(Eq <=> 0);
  ASSERT_SAME_TYPE(decltype(Eq <=> 0), std::strong_ordering);
  ASSERT_SAME_TYPE(decltype(0 <=> Eq), std::strong_ordering);
#endif
}

constexpr void test_equality() {
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto& StrongEq = std::strong_ordering::equal;
  auto& PartialEq = std::partial_ordering::equivalent;
  assert(StrongEq == PartialEq);

  auto& WeakEq = std::weak_ordering::equivalent;
  assert(StrongEq == WeakEq);
#endif
}

constexpr bool test_conversion() {
  static_assert(std::is_convertible<const std::strong_ordering&,
      std::partial_ordering>::value, "");
  { // value == 0
    auto V = std::strong_ordering::equivalent;
    std::partial_ordering WV = V;
    assert(WV == 0);
  }
  { // value < 0
    auto V = std::strong_ordering::less;
    std::partial_ordering WV = V;
    assert(WV < 0);
  }
  { // value > 0
    auto V = std::strong_ordering::greater;
    std::partial_ordering WV = V;
    assert(WV > 0);
  }

  static_assert(std::is_convertible<const std::strong_ordering&,
      std::weak_ordering>::value, "");
  { // value == 0
    auto V = std::strong_ordering::equivalent;
    std::weak_ordering WV = V;
    assert(WV == 0);
  }
  { // value < 0
    auto V = std::strong_ordering::less;
    std::weak_ordering WV = V;
    assert(WV < 0);
  }
  { // value > 0
    auto V = std::strong_ordering::greater;
    std::weak_ordering WV = V;
    assert(WV > 0);
  }
  return true;
}

constexpr bool test_constexpr() {
  auto& Eq = std::strong_ordering::equal;
  auto& Equiv = std::strong_ordering::equivalent;
  auto& Less = std::strong_ordering::less;
  auto& Greater = std::strong_ordering::greater;
  struct {
    std::strong_ordering Value;
    bool ExpectEq;
    bool ExpectNeq;
    bool ExpectLess;
    bool ExpectGreater;
  } TestCases[] = {
      {Eq, true, false, false, false},
      {Equiv, true, false, false, false},
      {Less, false, true, true, false},
      {Greater, false, true, false, true},
  };
  for (auto TC : TestCases) {
    auto V = TC.Value;
    assert((V == 0) == TC.ExpectEq);
    assert((0 == V) == TC.ExpectEq);
    assert((V != 0) == TC.ExpectNeq);
    assert((0 != V) == TC.ExpectNeq);

    assert((V < 0) == TC.ExpectLess);
    assert((V > 0) == TC.ExpectGreater);
    assert((V <= 0) == (TC.ExpectLess || TC.ExpectEq));
    assert((V >= 0) == (TC.ExpectGreater || TC.ExpectEq));

    assert((0 < V) == TC.ExpectGreater);
    assert((0 > V) == TC.ExpectLess);
    assert((0 <= V) == (TC.ExpectGreater || TC.ExpectEq));
    assert((0 >= V) == (TC.ExpectLess || TC.ExpectEq));
  }
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  {
    std::strong_ordering res = (Eq <=> 0);
    ((void)res);
    res = (0 <=> Eq);
    ((void)res);
  }
  enum ExpectRes {
    ER_Greater,
    ER_Less,
    ER_Equiv
  };
  struct {
    std::strong_ordering Value;
    ExpectRes Expect;
  } SpaceshipTestCases[] = {
      {std::strong_ordering::equivalent, ER_Equiv},
      {std::strong_ordering::less, ER_Less},
      {std::strong_ordering::greater, ER_Greater},
  };
  for (auto TC : SpaceshipTestCases)
  {
    std::strong_ordering Res = (TC.Value <=> 0);
    switch (TC.Expect) {
    case ER_Equiv:
      assert(Res == 0);
      assert(0 == Res);
      break;
    case ER_Less:
      assert(Res < 0);
      break;
    case ER_Greater:
      assert(Res > 0);
      break;
    }
  }
  {
    static_assert(std::strong_ordering::less == std::strong_ordering::less);
    static_assert(std::strong_ordering::less != std::strong_ordering::equal);
    static_assert(std::strong_ordering::less != std::strong_ordering::greater);

    static_assert(std::strong_ordering::equal != std::strong_ordering::less);
    static_assert(std::strong_ordering::equal == std::strong_ordering::equal);
    static_assert(std::strong_ordering::equal != std::strong_ordering::greater);

    static_assert(std::strong_ordering::greater != std::strong_ordering::less);
    static_assert(std::strong_ordering::greater != std::strong_ordering::equal);
    static_assert(std::strong_ordering::greater ==
                  std::strong_ordering::greater);
  }

  test_equality();
#endif

  return true;
}

int main(int, char**) {
  test_static_members();
  test_signatures();
  test_equality();
  static_assert(test_conversion(), "conversion test failed");
  static_assert(test_constexpr(), "constexpr test failed");

  return 0;
}
