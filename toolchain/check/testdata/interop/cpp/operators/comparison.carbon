// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/comparison.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/comparison.carbon

// --- test_rules.h

namespace std {
  class strong_ordering {
  public:
    static const strong_ordering less;
    static const strong_ordering equal;
    static const strong_ordering equivalent;
    static const strong_ordering greater;
  private:
    int value_;

    constexpr explicit strong_ordering(int x) noexcept : value_(x) {}
  };

  constexpr auto strong_ordering::less = strong_ordering(-1);
  constexpr auto strong_ordering::equal = strong_ordering(0);
  constexpr auto strong_ordering::equivalent = strong_ordering(0);
  constexpr auto strong_ordering::greater = strong_ordering(1);

  template<class From, class To>
  concept convertible_to =
    __is_convertible(From, To) &&
    requires(From from) {
      static_cast<To>(static_cast<From&&>(from));
    };
}

namespace TestRules {
  template<class T, class U>
  concept EqWith =
    requires(const T& x, const U& y) {
      // We relax `x == y` to return something that's convertible to `bool` as
      // C++ comparisons permit that. Carbon will eventually need to support
      // this in its interop.
      { x == y } -> std::convertible_to<bool>;
      { x != y } -> std::convertible_to<bool>;
    };
}

// --- returns_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct BothMembers {
  auto operator==(BothMembers const&) const -> bool;
  auto operator!=(BothMembers const&) const -> bool;
};
static_assert(TestRules::EqWith<BothMembers, BothMembers>);

struct MemberEqHiddenFriendNeq {
  auto operator==(MemberEqHiddenFriendNeq) const -> bool;

  friend auto operator!=(MemberEqHiddenFriendNeq x, MemberEqHiddenFriendNeq y) -> bool {
    return !(x == y);
  }
};
static_assert(TestRules::EqWith<MemberEqHiddenFriendNeq, MemberEqHiddenFriendNeq>);

struct HiddenFirendEqMemberNeq {
  auto operator!=(HiddenFirendEqMemberNeq) const -> bool;

  friend auto operator==(HiddenFirendEqMemberNeq x, HiddenFirendEqMemberNeq y) -> bool {
    return !(x != y);
  }
};
static_assert(TestRules::EqWith<MemberEqHiddenFriendNeq, MemberEqHiddenFriendNeq>);

struct BothHiddenFriends {
  friend auto operator==(BothHiddenFriends x, BothHiddenFriends y) -> bool {
    return true;
  }

  friend auto operator!=(BothHiddenFriends x, BothHiddenFriends y) -> bool {
    return false;
  }
};
static_assert(TestRules::EqWith<MemberEqHiddenFriendNeq, MemberEqHiddenFriendNeq>);

struct MemberEqFunctionNeq {
  auto operator==(MemberEqFunctionNeq) const -> bool;
};
auto operator!=(MemberEqFunctionNeq, MemberEqFunctionNeq) -> bool;
static_assert(TestRules::EqWith<MemberEqFunctionNeq, MemberEqFunctionNeq>);

struct FunctionEqMemberNeq {
  auto operator!=(FunctionEqMemberNeq) const -> bool;
};
auto operator==(FunctionEqMemberNeq, FunctionEqMemberNeq) -> bool;
static_assert(TestRules::EqWith<FunctionEqMemberNeq, FunctionEqMemberNeq>);


struct BothFunctions {};
auto operator==(BothFunctions, BothFunctions) -> bool;
auto operator!=(BothFunctions, BothFunctions) -> bool;
static_assert(TestRules::EqWith<BothFunctions, BothFunctions>);
''';

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(a: Cpp.BothMembers,
        b: Cpp.MemberEqHiddenFriendNeq,
        c: Cpp.HiddenFirendEqMemberNeq,
        d: Cpp.BothHiddenFriends,
        e: Cpp.MemberEqFunctionNeq,
        f: Cpp.FunctionEqMemberNeq,
        g: Cpp.BothFunctions) {
  EqWith(a, a);
  EqWith(b, b);
  EqWith(c, c);
  EqWith(d, d);
  EqWith(e, e);
  EqWith(f, f);
  EqWith(g, g);
}

// --- cross_type_equality.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct B1;
struct A1 {
  auto operator==(B1) const -> bool;
  auto operator!=(B1) const -> bool;
};
struct B1 {
  auto operator==(A1) const -> bool;
  auto operator!=(A1) const -> bool;
};
static_assert(TestRules::EqWith<A1, B1>);
static_assert(TestRules::EqWith<B1, A1>);

struct B2;
struct A2 {
};
struct B2 {
  auto operator==(A2) const -> bool;
  auto operator!=(A2) const -> bool;
};
static_assert(!TestRules::EqWith<A2, B2>);
static_assert(TestRules::EqWith<B2, A2>);
''';

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(a1: Cpp.A1, b1: Cpp.B1,
        a2: Cpp.A2, b2: Cpp.B2) {
  EqWith(a1, b1);
  EqWith(b1, a1);

  EqWith(b2, a2);
}

// --- fail_todo_returns_boolish.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct EqBoolNeqBoolish {
  auto operator==(EqBoolNeqBoolish) const -> bool;
  auto operator!=(EqBoolNeqBoolish) const -> int;
};
static_assert(TestRules::EqWith<EqBoolNeqBoolish, EqBoolNeqBoolish>);

struct EqBoolishNeqBool {
  auto operator==(EqBoolishNeqBool) const -> int*;
  auto operator!=(EqBoolishNeqBool) const -> bool;
};
static_assert(TestRules::EqWith<EqBoolishNeqBool, EqBoolishNeqBool>);

struct BothBoolish {
  enum Result {
    False,
    True,
  };
  auto operator==(BothBoolish) const -> Result;
  auto operator!=(BothBoolish) const -> Result;
};
static_assert(TestRules::EqWith<BothBoolish, BothBoolish>);
''';

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(a: Cpp.EqBoolNeqBoolish,
        b: Cpp.EqBoolishNeqBool,
        c: Cpp.BothBoolish) {
  EqWith(a, a);

  EqWith(b, b);

  EqWith(c, c);
}

// --- fail_todo_comparison_rewrites.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct DefaultEqOnly {
public:
  auto operator==(const DefaultEqOnly&) const -> bool = default;
};
static_assert(TestRules::EqWith<DefaultEqOnly, DefaultEqOnly>);

struct DefaultSpaceshipOnly {
public:
  auto operator<=>(const DefaultSpaceshipOnly&) const = default;
};
static_assert(TestRules::EqWith<DefaultSpaceshipOnly, DefaultSpaceshipOnly>);

struct OnlyEqDeclared {
public:
  auto operator==(OnlyEqDeclared) const -> bool;
};
static_assert(TestRules::EqWith<OnlyEqDeclared, OnlyEqDeclared>);
''';

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(a: Cpp.DefaultEqOnly,
        b: Cpp.DefaultSpaceshipOnly,
        c: Cpp.OnlyEqDeclared) {
  EqWith(a, a);

  EqWith(b, b);

  EqWith(c, c);
}

// --- fail_never_valid.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct OnlyNeq {
  auto operator!=(OnlyNeq) const -> bool;
};
static_assert(!TestRules::EqWith<OnlyNeq, OnlyNeq>);

struct OnlyNonDefaultSpaceship {
  auto operator<=>(OnlyNonDefaultSpaceship) const -> std::strong_ordering;
};
static_assert(!TestRules::EqWith<OnlyNonDefaultSpaceship, OnlyNonDefaultSpaceship>);

struct NoRelevantMembers {};
static_assert(!TestRules::EqWith<NoRelevantMembers, NoRelevantMembers>);

struct A2 {};
struct B2 {
  auto operator==(A2) const -> bool;
  auto operator!=(A2) const -> bool;
};
static_assert(!TestRules::EqWith<B2, B2>);
static_assert(TestRules::EqWith<B2, A2>);
static_assert(!TestRules::EqWith<A2, B2>);
''';

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(x: Cpp.OnlyNeq,
        y: Cpp.OnlyNonDefaultSpaceship,
        z: Cpp.NoRelevantMembers,
        a2: Cpp.A2, b2: Cpp.B2) {
  EqWith(x, x);

  EqWith(y, y);

  EqWith(z, z);

  EqWith(b2, b2);

  EqWith(a2, b2);
}
