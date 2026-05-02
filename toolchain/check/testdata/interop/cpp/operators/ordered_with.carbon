// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/ordered_with.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/ordered_with.carbon

// --- test_rules.h

namespace std {
  class partial_ordering {
  public:
    static const partial_ordering less;
    static const partial_ordering equivalent;
    static const partial_ordering greater;
    static const partial_ordering unordered;

    friend constexpr bool operator==(partial_ordering v, int) noexcept;
    friend constexpr bool operator==(partial_ordering v, partial_ordering w) noexcept = default;
    friend constexpr bool operator<(partial_ordering v, int) noexcept;
    friend constexpr bool operator>(partial_ordering v, int) noexcept;
    friend constexpr bool operator<=(partial_ordering v, int) noexcept;
    friend constexpr bool operator>=(partial_ordering v, int) noexcept;
    friend constexpr bool operator<(int, partial_ordering v) noexcept;
    friend constexpr bool operator>(int, partial_ordering v) noexcept;
    friend constexpr bool operator<=(int, partial_ordering v) noexcept;
    friend constexpr bool operator>=(int, partial_ordering v) noexcept;
    friend constexpr partial_ordering operator<=>(partial_ordering v, int) noexcept;
    friend constexpr partial_ordering operator<=>(int, partial_ordering v) noexcept;
  private:
    int value_;
    constexpr explicit partial_ordering(int x) noexcept : value_(x) {}
  };
  constexpr auto partial_ordering::less = partial_ordering(-1);
  constexpr auto partial_ordering::equivalent = partial_ordering(0);
  constexpr auto partial_ordering::greater = partial_ordering(1);
  constexpr auto partial_ordering::unordered = partial_ordering(2);

  class weak_ordering {
  public:
    static const weak_ordering less;
    static const weak_ordering equivalent;
    static const weak_ordering greater;

    friend constexpr bool operator==(weak_ordering v, int) noexcept;
    friend constexpr bool operator==(weak_ordering v, weak_ordering w) noexcept = default;
    friend constexpr bool operator<(weak_ordering v, int) noexcept;
    friend constexpr bool operator>(weak_ordering v, int) noexcept;
    friend constexpr bool operator<=(weak_ordering v, int) noexcept;
    friend constexpr bool operator>=(weak_ordering v, int) noexcept;
    friend constexpr bool operator<(int, weak_ordering v) noexcept;
    friend constexpr bool operator>(int, weak_ordering v) noexcept;
    friend constexpr bool operator<=(int, weak_ordering v) noexcept;
    friend constexpr bool operator>=(int, weak_ordering v) noexcept;
    friend constexpr weak_ordering operator<=>(weak_ordering v, int) noexcept;
    friend constexpr weak_ordering operator<=>(int, weak_ordering v) noexcept;
  private:
    int value_;
    constexpr explicit weak_ordering(int x) noexcept : value_(x) {}
  };
  constexpr auto weak_ordering::less = weak_ordering(-1);
  constexpr auto weak_ordering::equivalent = weak_ordering(0);
  constexpr auto weak_ordering::greater = weak_ordering(1);

  class strong_ordering {
  public:
    static const strong_ordering less;
    static const strong_ordering equal;
    static const strong_ordering equivalent;
    static const strong_ordering greater;

    friend constexpr bool operator==(strong_ordering v, int) noexcept;
    friend constexpr bool operator==(strong_ordering v, strong_ordering w) noexcept = default;
    friend constexpr bool operator<(strong_ordering v, int) noexcept;
    friend constexpr bool operator>(strong_ordering v, int) noexcept;
    friend constexpr bool operator<=(strong_ordering v, int) noexcept;
    friend constexpr bool operator>=(strong_ordering v, int) noexcept;
    friend constexpr bool operator<(int, strong_ordering v) noexcept;
    friend constexpr bool operator>(int, strong_ordering v) noexcept;
    friend constexpr bool operator<=(int, strong_ordering v) noexcept;
    friend constexpr bool operator>=(int, strong_ordering v) noexcept;
    friend constexpr strong_ordering operator<=>(strong_ordering v, int) noexcept;
    friend constexpr strong_ordering operator<=>(int, strong_ordering v) noexcept;
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
  concept OrderedWith =
    requires(const T& x, const U& y) {
      { x < y } -> std::convertible_to<bool>;
      { x <= y } -> std::convertible_to<bool>;
      { x > y } -> std::convertible_to<bool>;
      { x >= y } -> std::convertible_to<bool>;
    };
}

// --- returns_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct AllMembers {
  auto operator<(AllMembers const&) const -> bool;
  auto operator<=(AllMembers const&) const -> bool;
  auto operator>(AllMembers const&) const -> bool;
  auto operator>=(AllMembers const&) const -> bool;
};
static_assert(TestRules::OrderedWith<AllMembers, AllMembers>);

struct MixedMembersAndFriends {
  auto operator<(MixedMembersAndFriends const&) const -> bool;
  auto operator<=(MixedMembersAndFriends const&) const -> bool;

  friend auto operator>(MixedMembersAndFriends, MixedMembersAndFriends) -> bool { return true; }
  friend auto operator>=(MixedMembersAndFriends, MixedMembersAndFriends) -> bool { return true; }
};
static_assert(TestRules::OrderedWith<MixedMembersAndFriends, MixedMembersAndFriends>);

struct AllHiddenFriends {
  friend auto operator<(AllHiddenFriends, AllHiddenFriends) -> bool { return true; }
  friend auto operator<=(AllHiddenFriends, AllHiddenFriends) -> bool { return true; }
  friend auto operator>(AllHiddenFriends, AllHiddenFriends) -> bool { return true; }
  friend auto operator>=(AllHiddenFriends, AllHiddenFriends) -> bool { return true; }
};
static_assert(TestRules::OrderedWith<AllHiddenFriends, AllHiddenFriends>);

struct AllFunctions {};
auto operator<(AllFunctions, AllFunctions) -> bool;
auto operator<=(AllFunctions, AllFunctions) -> bool;
auto operator>(AllFunctions, AllFunctions) -> bool;
auto operator>=(AllFunctions, AllFunctions) -> bool;
static_assert(TestRules::OrderedWith<AllFunctions, AllFunctions>);
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x < y;
  x <= y;
  x > y;
  x >= y;
  //@dump-sem-ir-end
}

fn Test(all_members: Cpp.AllMembers,
        mixed_members_and_friends: Cpp.MixedMembersAndFriends,
        all_hidden_friends: Cpp.AllHiddenFriends,
        all_functions: Cpp.AllFunctions) {
  OrderedWith(all_members, all_members);
  OrderedWith(mixed_members_and_friends, mixed_members_and_friends);
  OrderedWith(all_hidden_friends, all_hidden_friends);
  OrderedWith(all_functions, all_functions);
}

// --- cross_type_ordering.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct B1;
struct A1 {
  auto operator<(B1) const -> bool;
  auto operator<=(B1) const -> bool;
  auto operator>(B1) const -> bool;
  auto operator>=(B1) const -> bool;
};
struct B1 {
  auto operator<(A1) const -> bool;
  auto operator<=(A1) const -> bool;
  auto operator>(A1) const -> bool;
  auto operator>=(A1) const -> bool;
};
static_assert(TestRules::OrderedWith<A1, B1>);
static_assert(TestRules::OrderedWith<B1, A1>);

struct B2;
struct A2 {
};
struct B2 {
  auto operator<(A2) const -> bool;
  auto operator<=(A2) const -> bool;
  auto operator>(A2) const -> bool;
  auto operator>=(A2) const -> bool;
};
static_assert(!TestRules::OrderedWith<A2, B2>);
static_assert(TestRules::OrderedWith<B2, A2>);

struct B3;
struct A3 {
  auto operator<(B3 b3) const -> bool;
  auto operator>(B3 b3) const -> bool;

  friend auto operator<(B3 const& b3, A3 a3) -> bool {
    return true;
  }
  friend auto operator>(B3 const& b3, A3 a3) -> bool {
    return false;
  }
};
struct B3 {
  auto operator<=(A3 a3) const -> bool;
  auto operator>=(A3 a3) const -> bool;

  friend auto operator<=(A3 const& a3, B3 b3) -> bool {
    return false;
  }

  friend auto operator>=(A3 const& a3, B3 b3) -> bool {
    return true;
  }
};
static_assert(TestRules::OrderedWith<A3, B3>);
static_assert(TestRules::OrderedWith<B3, A3>);
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x < y;
  x <= y;
  x > y;
  x >= y;
  //@dump-sem-ir-end
}

fn Test(a1: Cpp.A1, b1: Cpp.B1,
        a2: Cpp.A2, b2: Cpp.B2,
        a3: Cpp.A3, b3: Cpp.B3) {
  OrderedWith(a1, b1);
  OrderedWith(b1, a1);

  OrderedWith(b2, a2);

  OrderedWith(a3, b3);
  OrderedWith(b3, a3);
}

// --- fail_todo_returns_boolish.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct LessReturnsBoolish {
  auto operator<(LessReturnsBoolish) const -> int;
  auto operator<=(LessReturnsBoolish) const -> bool;
  auto operator>(LessReturnsBoolish) const -> bool;
  auto operator>=(LessReturnsBoolish) const -> bool;
};
static_assert(TestRules::OrderedWith<LessReturnsBoolish, LessReturnsBoolish>);

struct LessEqualReturnsBoolish {
  auto operator<(LessEqualReturnsBoolish) const -> bool;
  auto operator<=(LessEqualReturnsBoolish) const -> int*;
  auto operator>(LessEqualReturnsBoolish) const -> bool;
  auto operator>=(LessEqualReturnsBoolish) const -> bool;
};
static_assert(TestRules::OrderedWith<LessEqualReturnsBoolish, LessEqualReturnsBoolish>);

struct GreaterEqualReturnsBoolish {
  enum Result { False, True };
  auto operator<(GreaterEqualReturnsBoolish) const -> bool;
  auto operator<=(GreaterEqualReturnsBoolish) const -> bool;
  auto operator>(GreaterEqualReturnsBoolish) const -> bool;
  auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
};
static_assert(TestRules::OrderedWith<GreaterEqualReturnsBoolish, GreaterEqualReturnsBoolish>);

struct GreaterReturnsBoolish {
  struct BoolLike {
    BoolLike(bool);
    operator bool() const;
  };
  auto operator<(GreaterReturnsBoolish) const -> bool;
  auto operator<=(GreaterReturnsBoolish) const -> bool;
  auto operator>(GreaterReturnsBoolish) const -> BoolLike;
  auto operator>=(GreaterReturnsBoolish) const -> bool;
};
static_assert(TestRules::OrderedWith<GreaterReturnsBoolish, GreaterReturnsBoolish>);

struct AllReturnBoolish {
  auto operator<(AllReturnBoolish) const -> double;
  auto operator<=(AllReturnBoolish) const -> double;
  auto operator>(AllReturnBoolish) const -> double;
  auto operator>=(AllReturnBoolish) const -> double;
};
static_assert(TestRules::OrderedWith<AllReturnBoolish, AllReturnBoolish>);
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  x < y;
  x <= y;
  x > y;
  x >= y;
}

fn Test(less_returns_boolish: Cpp.LessReturnsBoolish,
        less_equal_returns_boolish: Cpp.LessEqualReturnsBoolish,
        greater_equal_returns_boolish: Cpp.GreaterEqualReturnsBoolish,
        greater_returns_boolish: Cpp.GreaterReturnsBoolish,
        all_return_boolish: Cpp.AllReturnBoolish) {
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+14]]:3: error: semantics TODO: `Comparison operations must return `bool`` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(less_returns_boolish, less_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.LessReturnsBoolish` into type implementing `Core.OrderedWith(Cpp.LessReturnsBoolish)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(less_returns_boolish, less_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-22]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_returns_boolish, less_returns_boolish);
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+14]]:3: error: semantics TODO: `Comparison operations must return `bool`` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(less_equal_returns_boolish, less_equal_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-30]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.LessEqualReturnsBoolish` into type implementing `Core.OrderedWith(Cpp.LessEqualReturnsBoolish)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(less_equal_returns_boolish, less_equal_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-37]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_equal_returns_boolish, less_equal_returns_boolish);
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+14]]:3: error: semantics TODO: `Comparison operations must return `bool`` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(greater_equal_returns_boolish, greater_equal_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-45]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.GreaterEqualReturnsBoolish` into type implementing `Core.OrderedWith(Cpp.GreaterEqualReturnsBoolish)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(greater_equal_returns_boolish, greater_equal_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(greater_equal_returns_boolish, greater_equal_returns_boolish);
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+14]]:3: error: semantics TODO: `Comparison operations must return `bool`` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(greater_returns_boolish, greater_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-60]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.GreaterReturnsBoolish` into type implementing `Core.OrderedWith(Cpp.GreaterReturnsBoolish)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(greater_returns_boolish, greater_returns_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-67]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(greater_returns_boolish, greater_returns_boolish);
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+14]]:3: error: semantics TODO: `Comparison operations must return `bool`` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(all_return_boolish, all_return_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-75]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.AllReturnBoolish` into type implementing `Core.OrderedWith(Cpp.AllReturnBoolish)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(all_return_boolish, all_return_boolish);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-82]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(all_return_boolish, all_return_boolish);
}

// --- fail_todo_comparison_rewrites.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct DefaultSpaceshipOnly {
public:
  auto operator<=>(const DefaultSpaceshipOnly&) const = default;
};
static_assert(TestRules::OrderedWith<DefaultSpaceshipOnly, DefaultSpaceshipOnly>);

struct ReturnsStrongOrdering {
  auto operator<=>(ReturnsStrongOrdering) const -> std::strong_ordering;
};
static_assert(TestRules::OrderedWith<ReturnsStrongOrdering, ReturnsStrongOrdering>);

struct ReturnsWeakOrdering {
  auto operator<=>(ReturnsWeakOrdering) const -> std::weak_ordering;
};
static_assert(TestRules::OrderedWith<ReturnsWeakOrdering, ReturnsWeakOrdering>);

struct ReturnsPartialOrdering {
  auto operator<=>(ReturnsPartialOrdering) const -> std::partial_ordering;
};
static_assert(TestRules::OrderedWith<ReturnsPartialOrdering, ReturnsPartialOrdering>);

struct B4;
struct A4 {
  auto operator<=>(B4) const -> std::strong_ordering;
};
struct B4 {
  auto operator<=>(A4) const -> std::strong_ordering;
};
static_assert(TestRules::OrderedWith<A4, B4>);
static_assert(TestRules::OrderedWith<B4, A4>);
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  x < y;
  x <= y;
  x > y;
  x >= y;
}

fn Test(default_spaceship_only: Cpp.DefaultSpaceshipOnly,
        returns_strong_ordering: Cpp.ReturnsStrongOrdering,
        returns_weak_ordering: Cpp.ReturnsWeakOrdering,
        returns_partial_ordering: Cpp.ReturnsPartialOrdering,
        a4: Cpp.A4, b4: Cpp.B4) {
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(default_spaceship_only, default_spaceship_only);
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-23]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_strong_ordering, returns_strong_ordering);
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_weak_ordering, returns_weak_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-31]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_weak_ordering, returns_weak_ordering);
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(returns_partial_ordering, returns_partial_ordering);
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(a4, b4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-47]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(a4, b4);
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator< using operator<=> is not supported` [SemanticsTodo]
  // CHECK:STDERR:   OrderedWith(b4, a4);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-55]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(b4, a4);
}

// --- fail_never_valid.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
#include "test_rules.h"

struct MissingLess {
  auto operator<=(MissingLess) const -> bool;
  auto operator>(MissingLess) const -> bool;
  auto operator>=(MissingLess) const -> bool;
};
static_assert(!TestRules::OrderedWith<MissingLess, MissingLess>);

struct MissingLessEqual {
  auto operator<(MissingLessEqual) const -> bool;
  auto operator>(MissingLessEqual) const -> bool;
  auto operator>=(MissingLessEqual) const -> bool;
};
static_assert(!TestRules::OrderedWith<MissingLessEqual, MissingLessEqual>);

struct MissingGreater {
  auto operator<(MissingGreater) const -> bool;
  auto operator<=(MissingGreater) const -> bool;
  auto operator>=(MissingGreater) const -> bool;
};
static_assert(!TestRules::OrderedWith<MissingGreater, MissingGreater>);

struct MissingGreaterEqual {
  auto operator<(MissingGreaterEqual) const -> bool;
  auto operator<=(MissingGreaterEqual) const -> bool;
  auto operator>(MissingGreaterEqual) const -> bool;
};
static_assert(!TestRules::OrderedWith<MissingGreaterEqual, MissingGreaterEqual>);

struct NoRelevantMembers {};
static_assert(!TestRules::OrderedWith<NoRelevantMembers, NoRelevantMembers>);

struct LessReturnsVoid {
  auto operator<(LessReturnsVoid) const -> void;
  auto operator<=(LessReturnsVoid) const -> bool;
  auto operator>(LessReturnsVoid) const -> bool;
  auto operator>=(LessReturnsVoid) const -> bool;
};
static_assert(!TestRules::OrderedWith<LessReturnsVoid, LessReturnsVoid>);

struct LessNonConst {
  auto operator<(LessNonConst) -> bool;
  auto operator<=(LessNonConst) const -> bool;
  auto operator>(LessNonConst) const -> bool;
  auto operator>=(LessNonConst) const -> bool;
};
static_assert(!TestRules::OrderedWith<LessNonConst, LessNonConst>);
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  x < y;
  x <= y;
  x > y;
  x >= y;
}

fn Test(missing_less: Cpp.MissingLess,
        missing_less_equal: Cpp.MissingLessEqual,
        missing_greater: Cpp.MissingGreater,
        missing_greater_equal: Cpp.MissingGreaterEqual,
        no_relevant_members: Cpp.NoRelevantMembers,
        less_returns_void: Cpp.LessReturnsVoid) {
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingLess` into type implementing `Core.OrderedWith(Cpp.MissingLess)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_less, missing_less);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-16]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less, missing_less);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingLessEqual` into type implementing `Core.OrderedWith(Cpp.MissingLessEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_less_equal, missing_less_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-25]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less_equal, missing_less_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreater` into type implementing `Core.OrderedWith(Cpp.MissingGreater)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater, missing_greater);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-34]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater, missing_greater);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreaterEqual` into type implementing `Core.OrderedWith(Cpp.MissingGreaterEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater_equal, missing_greater_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater_equal, missing_greater_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoRelevantMembers` into type implementing `Core.OrderedWith(Cpp.NoRelevantMembers)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(no_relevant_members, no_relevant_members);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(no_relevant_members, no_relevant_members);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.LessReturnsVoid` into type implementing `Core.OrderedWith(Cpp.LessReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(less_returns_void, less_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-61]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_returns_void, less_returns_void);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+8]]:15: error: name `less_non_const` not found [NameNotFound]
  // CHECK:STDERR:   OrderedWith(less_non_const, less_non_const);
  // CHECK:STDERR:               ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+4]]:31: error: name `less_non_const` not found [NameNotFound]
  // CHECK:STDERR:   OrderedWith(less_non_const, less_non_const);
  // CHECK:STDERR:                               ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_non_const, less_non_const);
}

// CHECK:STDOUT: --- returns_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ca8786.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.adc9f7.1: %OrderedWith.type.ca8786.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2b366e.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.975556.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6f28f1.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c82b75.1: %OrderedWith.WithSelf.Greater.type.6f28f1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.970504.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.86efc4.1: %OrderedWith.WithSelf.LessOrEquivalent.type.970504.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.df27e9.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.1a4163.1: %OrderedWith.WithSelf.Less.type.df27e9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ca8786.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.d3f: type = pattern_type %OrderedWith.type.ca8786.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.ca8786.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2b366e.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.f6e965.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element0, imports.%Core.import_ref.255 [symbolic]
// CHECK:STDOUT:   %assoc1.23402c.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element1, imports.%Core.import_ref.857 [symbolic]
// CHECK:STDOUT:   %assoc2.771acd.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic]
// CHECK:STDOUT:   %assoc3.ba2aca.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.c70b34.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.3d2: <witness> = require_complete_type %OrderedWith.type.ca8786.2 [symbolic]
// CHECK:STDOUT:   %assoc0.666: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element0, imports.%Core.import_ref.e2b [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.df27e9.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.970504.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6f28f1.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.0af: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.df27e9.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0af = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.072: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.e40: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element1, imports.%Core.import_ref.ce4 [symbolic]
// CHECK:STDOUT:   %.81b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.970504.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.81b = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f8: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.b0e: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element2, imports.%Core.import_ref.40e [symbolic]
// CHECK:STDOUT:   %.f69: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.6f28f1.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.f69 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1cd: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.078: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element3, imports.%Core.import_ref.a50 [symbolic]
// CHECK:STDOUT:   %.206: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.206 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b2f: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %AllMembers: type = class_type @AllMembers [concrete]
// CHECK:STDOUT:   %pattern_type.31d: type = pattern_type %AllMembers [concrete]
// CHECK:STDOUT:   %MixedMembersAndFriends: type = class_type @MixedMembersAndFriends [concrete]
// CHECK:STDOUT:   %pattern_type.296: type = pattern_type %MixedMembersAndFriends [concrete]
// CHECK:STDOUT:   %AllHiddenFriends: type = class_type @AllHiddenFriends [concrete]
// CHECK:STDOUT:   %pattern_type.cae: type = pattern_type %AllHiddenFriends [concrete]
// CHECK:STDOUT:   %AllFunctions: type = class_type @AllFunctions [concrete]
// CHECK:STDOUT:   %pattern_type.a19: type = pattern_type %AllFunctions [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.45f: type = facet_type <@OrderedWith.1, @OrderedWith.1(%AllMembers)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.d7a: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%AllMembers) [concrete]
// CHECK:STDOUT:   %assoc0.eb2: %OrderedWith.assoc_type.d7a = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.aae: %OrderedWith.assoc_type.d7a = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.190: %OrderedWith.assoc_type.d7a = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.a9a: %OrderedWith.assoc_type.d7a = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %AllMembers.Less.type: type = fn_type @AllMembers.Less [concrete]
// CHECK:STDOUT:   %AllMembers.Less: %AllMembers.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %AllMembers.LessOrEquivalent.type: type = fn_type @AllMembers.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %AllMembers.LessOrEquivalent: %AllMembers.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %AllMembers.Greater.type: type = fn_type @AllMembers.Greater [concrete]
// CHECK:STDOUT:   %AllMembers.Greater: %AllMembers.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %AllMembers.GreaterOrEquivalent.type: type = fn_type @AllMembers.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %AllMembers.GreaterOrEquivalent: %AllMembers.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.87d: <witness> = custom_witness (%AllMembers.Less, %AllMembers.LessOrEquivalent, %AllMembers.Greater, %AllMembers.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%AllMembers) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.2ef: %OrderedWith.type.45f = facet_value %AllMembers, (%custom_witness.87d) [concrete]
// CHECK:STDOUT:   %pattern_type.73d: type = pattern_type %OrderedWith.type.45f [concrete]
// CHECK:STDOUT:   %OrderedWith.type.cac: type = facet_type <@OrderedWith.1, @OrderedWith.1(%MixedMembersAndFriends)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.c54: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%MixedMembersAndFriends) [concrete]
// CHECK:STDOUT:   %assoc0.7c0: %OrderedWith.assoc_type.c54 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.905: %OrderedWith.assoc_type.c54 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.0be: %OrderedWith.assoc_type.c54 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.602: %OrderedWith.assoc_type.c54 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %MixedMembersAndFriends.Less.type: type = fn_type @MixedMembersAndFriends.Less [concrete]
// CHECK:STDOUT:   %MixedMembersAndFriends.Less: %MixedMembersAndFriends.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %MixedMembersAndFriends.LessOrEquivalent.type: type = fn_type @MixedMembersAndFriends.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %MixedMembersAndFriends.LessOrEquivalent: %MixedMembersAndFriends.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type.a27415.1: type = fn_type @Greater.1 [concrete]
// CHECK:STDOUT:   %Greater.6a2449.1: %Greater.type.a27415.1 = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.794505.1: type = fn_type @GreaterOrEquivalent.1 [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.a3b59c.1: %GreaterOrEquivalent.type.794505.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d9e: <witness> = custom_witness (%MixedMembersAndFriends.Less, %MixedMembersAndFriends.LessOrEquivalent, %Greater.6a2449.1, %GreaterOrEquivalent.a3b59c.1), @OrderedWith.1, @OrderedWith.1(%MixedMembersAndFriends) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fe6: %OrderedWith.type.cac = facet_value %MixedMembersAndFriends, (%custom_witness.d9e) [concrete]
// CHECK:STDOUT:   %pattern_type.d93: type = pattern_type %OrderedWith.type.cac [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9c5: type = facet_type <@OrderedWith.1, @OrderedWith.1(%AllHiddenFriends)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.643: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%AllHiddenFriends) [concrete]
// CHECK:STDOUT:   %assoc0.6c1: %OrderedWith.assoc_type.643 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.823: %OrderedWith.assoc_type.643 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.445: %OrderedWith.assoc_type.643 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.bba: %OrderedWith.assoc_type.643 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %Less.type.45ec63.1: type = fn_type @Less.1 [concrete]
// CHECK:STDOUT:   %Less.8f0ac4.1: %Less.type.45ec63.1 = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type.3ffaa7.1: type = fn_type @LessOrEquivalent.1 [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.9a8233.1: %LessOrEquivalent.type.3ffaa7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type.a27415.2: type = fn_type @Greater.2 [concrete]
// CHECK:STDOUT:   %Greater.6a2449.2: %Greater.type.a27415.2 = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.794505.2: type = fn_type @GreaterOrEquivalent.2 [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.a3b59c.2: %GreaterOrEquivalent.type.794505.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3c3: <witness> = custom_witness (%Less.8f0ac4.1, %LessOrEquivalent.9a8233.1, %Greater.6a2449.2, %GreaterOrEquivalent.a3b59c.2), @OrderedWith.1, @OrderedWith.1(%AllHiddenFriends) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.679: %OrderedWith.type.9c5 = facet_value %AllHiddenFriends, (%custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %pattern_type.a72: type = pattern_type %OrderedWith.type.9c5 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.549: type = facet_type <@OrderedWith.1, @OrderedWith.1(%AllFunctions)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.488: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%AllFunctions) [concrete]
// CHECK:STDOUT:   %assoc0.a0f: %OrderedWith.assoc_type.488 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.d6f: %OrderedWith.assoc_type.488 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.2fa: %OrderedWith.assoc_type.488 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.8e6: %OrderedWith.assoc_type.488 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %Less.type.45ec63.2: type = fn_type @Less.2 [concrete]
// CHECK:STDOUT:   %Less.8f0ac4.2: %Less.type.45ec63.2 = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type.3ffaa7.2: type = fn_type @LessOrEquivalent.2 [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.9a8233.2: %LessOrEquivalent.type.3ffaa7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type.a27415.3: type = fn_type @Greater.3 [concrete]
// CHECK:STDOUT:   %Greater.6a2449.3: %Greater.type.a27415.3 = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type.794505.3: type = fn_type @GreaterOrEquivalent.3 [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.a3b59c.3: %GreaterOrEquivalent.type.794505.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.63e: <witness> = custom_witness (%Less.8f0ac4.2, %LessOrEquivalent.9a8233.2, %Greater.6a2449.3, %GreaterOrEquivalent.a3b59c.3), @OrderedWith.1, @OrderedWith.1(%AllFunctions) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.ce9: %OrderedWith.type.549 = facet_value %AllFunctions, (%custom_witness.63e) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %OrderedWith.type.549 [concrete]
// CHECK:STDOUT:   %complete_type.f0d: <witness> = complete_type_witness %OrderedWith.type.45f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.a7e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%AllMembers, %OrderedWith.facet.2ef) [concrete]
// CHECK:STDOUT:   %.fd9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.a7e, %OrderedWith.facet.2ef [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.dc1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%AllMembers, %OrderedWith.facet.2ef) [concrete]
// CHECK:STDOUT:   %.561: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.dc1, %OrderedWith.facet.2ef [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.83f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%AllMembers, %OrderedWith.facet.2ef) [concrete]
// CHECK:STDOUT:   %.bad: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.83f, %OrderedWith.facet.2ef [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.346: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%AllMembers, %OrderedWith.facet.2ef) [concrete]
// CHECK:STDOUT:   %.619: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.346, %OrderedWith.facet.2ef [concrete]
// CHECK:STDOUT:   %complete_type.5e3: <witness> = complete_type_witness %OrderedWith.type.cac [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.71b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%MixedMembersAndFriends, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.dda: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.71b, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.b5e: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%MixedMembersAndFriends, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.324: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.b5e, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.cec: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%MixedMembersAndFriends, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.f24: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.cec, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.524: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%MixedMembersAndFriends, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.426: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.524, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %complete_type.520: <witness> = complete_type_witness %OrderedWith.type.9c5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.6a7: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%AllHiddenFriends, %OrderedWith.facet.679) [concrete]
// CHECK:STDOUT:   %.a14: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.6a7, %OrderedWith.facet.679 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.1e8: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%AllHiddenFriends, %OrderedWith.facet.679) [concrete]
// CHECK:STDOUT:   %.c1d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.1e8, %OrderedWith.facet.679 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9e7: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%AllHiddenFriends, %OrderedWith.facet.679) [concrete]
// CHECK:STDOUT:   %.ed8: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9e7, %OrderedWith.facet.679 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.442: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%AllHiddenFriends, %OrderedWith.facet.679) [concrete]
// CHECK:STDOUT:   %.3a7: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.442, %OrderedWith.facet.679 [concrete]
// CHECK:STDOUT:   %complete_type.5d1: <witness> = complete_type_witness %OrderedWith.type.549 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.26e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%AllFunctions, %OrderedWith.facet.ce9) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.26e, %OrderedWith.facet.ce9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.c81: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%AllFunctions, %OrderedWith.facet.ce9) [concrete]
// CHECK:STDOUT:   %.6b8: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.c81, %OrderedWith.facet.ce9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.898: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%AllFunctions, %OrderedWith.facet.ce9) [concrete]
// CHECK:STDOUT:   %.580: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.898, %OrderedWith.facet.ce9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.130: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%AllFunctions, %OrderedWith.facet.ce9) [concrete]
// CHECK:STDOUT:   %.cfa: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.130, %OrderedWith.facet.ce9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e9: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.666)]
// CHECK:STDOUT:   %Core.import_ref.c86: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.e40)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.b0e)]
// CHECK:STDOUT:   %Core.import_ref.065: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.078)]
// CHECK:STDOUT:   %Core.import_ref.bb2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.975556.1)]
// CHECK:STDOUT:   %Core.import_ref.a0a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.6f28f1.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c82b75.1)]
// CHECK:STDOUT:   %Core.import_ref.857: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.970504.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.86efc4.1)]
// CHECK:STDOUT:   %Core.import_ref.255: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.df27e9.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.1a4163.1)]
// CHECK:STDOUT:   %Core.import_ref.e2b = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ce4 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.40e = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a50 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc40(%U.loc40_17.2: type, %T.loc40_27.2: @OrderedWith.loc40.%OrderedWith.type.loc40_48.1 (%OrderedWith.type.ca8786.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc42: <witness> = require_complete_type %OrderedWith.type.loc40_48.1 [symbolic = %require_complete.loc42 (constants.%require_complete.3d2)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc40_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2b366e.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element0, imports.%Core.import_ref.255 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.df27e9.3)]
// CHECK:STDOUT:   %.loc42_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc40_27.1 [symbolic = %.loc42_5.2 (constants.%.0af)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc40_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc40_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc42_5.2: @OrderedWith.loc40.%.loc42_5.2 (%.0af) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc42_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_5.2: <specific function> = specific_impl_function %impl.elem0.loc42_5.2, @OrderedWith.WithSelf.Less(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %specific_impl_fn.loc42_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element1, imports.%Core.import_ref.857 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.970504.3)]
// CHECK:STDOUT:   %.loc43_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc40_27.1 [symbolic = %.loc43_5.2 (constants.%.81b)]
// CHECK:STDOUT:   %impl.elem1.loc43_5.2: @OrderedWith.loc40.%.loc43_5.2 (%.81b) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc43_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_5.2: <specific function> = specific_impl_function %impl.elem1.loc43_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %specific_impl_fn.loc43_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.6f28f1.3)]
// CHECK:STDOUT:   %.loc44_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc40_27.1 [symbolic = %.loc44_5.2 (constants.%.f69)]
// CHECK:STDOUT:   %impl.elem2.loc44_5.2: @OrderedWith.loc40.%.loc44_5.2 (%.f69) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_5.2: <specific function> = specific_impl_function %impl.elem2.loc44_5.2, @OrderedWith.WithSelf.Greater(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %specific_impl_fn.loc44_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3)]
// CHECK:STDOUT:   %.loc45_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc40_27.1 [symbolic = %.loc45_5.2 (constants.%.206)]
// CHECK:STDOUT:   %impl.elem3.loc45_5.2: @OrderedWith.loc40.%.loc45_5.2 (%.206) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_5.2: <specific function> = specific_impl_function %impl.elem3.loc45_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc40_17.1, %T.loc40_27.1) [symbolic = %specific_impl_fn.loc45_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc40.%T.as_type.loc40_54.1 (%T.as_type), %y.param: @OrderedWith.loc40.%U.loc40_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc42: @OrderedWith.loc40.%T.as_type.loc40_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc42: @OrderedWith.loc40.%U.loc40_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc42: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc40_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc42_5.1: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.7e9, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Less, %.loc42_5.1 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %impl.elem0.loc42_5.1: @OrderedWith.loc40.%.loc42_5.2 (%.0af) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc42_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc42_5.1: <bound method> = bound_method %x.ref.loc42, %impl.elem0.loc42_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_5.1: <specific function> = specific_impl_function %impl.elem0.loc42_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc42_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:     %bound_method.loc42_5.2: <bound method> = bound_method %x.ref.loc42, %specific_impl_fn.loc42_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc42_5.2(%x.ref.loc42, %y.ref.loc42)
// CHECK:STDOUT:     %x.ref.loc43: @OrderedWith.loc40.%T.as_type.loc40_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc43: @OrderedWith.loc40.%U.loc40_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc43: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc40_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc43_5.1: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.c86, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref LessOrEquivalent, %.loc43_5.1 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %impl.elem1.loc43_5.1: @OrderedWith.loc40.%.loc43_5.2 (%.81b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc43_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc43_5.1: <bound method> = bound_method %x.ref.loc43, %impl.elem1.loc43_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_5.1: <specific function> = specific_impl_function %impl.elem1.loc43_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc43_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc43_5.2: <bound method> = bound_method %x.ref.loc43, %specific_impl_fn.loc43_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc43_5.2(%x.ref.loc43, %y.ref.loc43)
// CHECK:STDOUT:     %x.ref.loc44: @OrderedWith.loc40.%T.as_type.loc40_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc44: @OrderedWith.loc40.%U.loc40_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc44: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc40_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc44_5.1: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.f4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Greater, %.loc44_5.1 [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %impl.elem2.loc44_5.1: @OrderedWith.loc40.%.loc44_5.2 (%.f69) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc44_5.1: <bound method> = bound_method %x.ref.loc44, %impl.elem2.loc44_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_5.1: <specific function> = specific_impl_function %impl.elem2.loc44_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc44_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:     %bound_method.loc44_5.2: <bound method> = bound_method %x.ref.loc44, %specific_impl_fn.loc44_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc44_5.2(%x.ref.loc44, %y.ref.loc44)
// CHECK:STDOUT:     %x.ref.loc45: @OrderedWith.loc40.%T.as_type.loc40_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc45: @OrderedWith.loc40.%U.loc40_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc45: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc40_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc45_5.1: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.065, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc40.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref GreaterOrEquivalent, %.loc45_5.1 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %impl.elem3.loc45_5.1: @OrderedWith.loc40.%.loc45_5.2 (%.206) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc45_5.1: <bound method> = bound_method %x.ref.loc45, %impl.elem3.loc45_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_5.1: <specific function> = specific_impl_function %impl.elem3.loc45_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc45_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:     %bound_method.loc45_5.2: <bound method> = bound_method %x.ref.loc45, %specific_impl_fn.loc45_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc45_5.2(%x.ref.loc45, %y.ref.loc45)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc40(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc40_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc40_48.1 => constants.%OrderedWith.type.ca8786.2
// CHECK:STDOUT:   %T.loc40_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc40_27 => constants.%pattern_type.d3f
// CHECK:STDOUT:   %T.as_type.loc40_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc40_52 => constants.%pattern_type.c70b34.2
// CHECK:STDOUT:   %pattern_type.loc40_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc40(constants.%AllMembers, constants.%OrderedWith.facet.2ef) {
// CHECK:STDOUT:   %U.loc40_17.1 => constants.%AllMembers
// CHECK:STDOUT:   %OrderedWith.type.loc40_48.1 => constants.%OrderedWith.type.45f
// CHECK:STDOUT:   %T.loc40_27.1 => constants.%OrderedWith.facet.2ef
// CHECK:STDOUT:   %pattern_type.loc40_27 => constants.%pattern_type.73d
// CHECK:STDOUT:   %T.as_type.loc40_54.1 => constants.%AllMembers
// CHECK:STDOUT:   %pattern_type.loc40_52 => constants.%pattern_type.31d
// CHECK:STDOUT:   %pattern_type.loc40_58 => constants.%pattern_type.31d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc40_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type.f0d
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.d7a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.eb2
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.a7e
// CHECK:STDOUT:   %.loc42_5.2 => constants.%.fd9
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.87d
// CHECK:STDOUT:   %impl.elem0.loc42_5.2 => constants.%AllMembers.Less
// CHECK:STDOUT:   %specific_impl_fn.loc42_5.2 => constants.%AllMembers.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.aae
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.dc1
// CHECK:STDOUT:   %.loc43_5.2 => constants.%.561
// CHECK:STDOUT:   %impl.elem1.loc43_5.2 => constants.%AllMembers.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc43_5.2 => constants.%AllMembers.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.190
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.83f
// CHECK:STDOUT:   %.loc44_5.2 => constants.%.bad
// CHECK:STDOUT:   %impl.elem2.loc44_5.2 => constants.%AllMembers.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc44_5.2 => constants.%AllMembers.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.a9a
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.346
// CHECK:STDOUT:   %.loc45_5.2 => constants.%.619
// CHECK:STDOUT:   %impl.elem3.loc45_5.2 => constants.%AllMembers.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc45_5.2 => constants.%AllMembers.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc40(constants.%MixedMembersAndFriends, constants.%OrderedWith.facet.fe6) {
// CHECK:STDOUT:   %U.loc40_17.1 => constants.%MixedMembersAndFriends
// CHECK:STDOUT:   %OrderedWith.type.loc40_48.1 => constants.%OrderedWith.type.cac
// CHECK:STDOUT:   %T.loc40_27.1 => constants.%OrderedWith.facet.fe6
// CHECK:STDOUT:   %pattern_type.loc40_27 => constants.%pattern_type.d93
// CHECK:STDOUT:   %T.as_type.loc40_54.1 => constants.%MixedMembersAndFriends
// CHECK:STDOUT:   %pattern_type.loc40_52 => constants.%pattern_type.296
// CHECK:STDOUT:   %pattern_type.loc40_58 => constants.%pattern_type.296
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc40_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type.5e3
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.c54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c0
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.71b
// CHECK:STDOUT:   %.loc42_5.2 => constants.%.dda
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.d9e
// CHECK:STDOUT:   %impl.elem0.loc42_5.2 => constants.%MixedMembersAndFriends.Less
// CHECK:STDOUT:   %specific_impl_fn.loc42_5.2 => constants.%MixedMembersAndFriends.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.905
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.b5e
// CHECK:STDOUT:   %.loc43_5.2 => constants.%.324
// CHECK:STDOUT:   %impl.elem1.loc43_5.2 => constants.%MixedMembersAndFriends.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc43_5.2 => constants.%MixedMembersAndFriends.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.0be
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.cec
// CHECK:STDOUT:   %.loc44_5.2 => constants.%.f24
// CHECK:STDOUT:   %impl.elem2.loc44_5.2 => constants.%Greater.6a2449.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_5.2 => constants.%Greater.6a2449.1
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.602
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.524
// CHECK:STDOUT:   %.loc45_5.2 => constants.%.426
// CHECK:STDOUT:   %impl.elem3.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc40(constants.%AllHiddenFriends, constants.%OrderedWith.facet.679) {
// CHECK:STDOUT:   %U.loc40_17.1 => constants.%AllHiddenFriends
// CHECK:STDOUT:   %OrderedWith.type.loc40_48.1 => constants.%OrderedWith.type.9c5
// CHECK:STDOUT:   %T.loc40_27.1 => constants.%OrderedWith.facet.679
// CHECK:STDOUT:   %pattern_type.loc40_27 => constants.%pattern_type.a72
// CHECK:STDOUT:   %T.as_type.loc40_54.1 => constants.%AllHiddenFriends
// CHECK:STDOUT:   %pattern_type.loc40_52 => constants.%pattern_type.cae
// CHECK:STDOUT:   %pattern_type.loc40_58 => constants.%pattern_type.cae
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc40_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type.520
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.643
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6c1
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.6a7
// CHECK:STDOUT:   %.loc42_5.2 => constants.%.a14
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.3c3
// CHECK:STDOUT:   %impl.elem0.loc42_5.2 => constants.%Less.8f0ac4.1
// CHECK:STDOUT:   %specific_impl_fn.loc42_5.2 => constants.%Less.8f0ac4.1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.823
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.1e8
// CHECK:STDOUT:   %.loc43_5.2 => constants.%.c1d
// CHECK:STDOUT:   %impl.elem1.loc43_5.2 => constants.%LessOrEquivalent.9a8233.1
// CHECK:STDOUT:   %specific_impl_fn.loc43_5.2 => constants.%LessOrEquivalent.9a8233.1
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.445
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.9e7
// CHECK:STDOUT:   %.loc44_5.2 => constants.%.ed8
// CHECK:STDOUT:   %impl.elem2.loc44_5.2 => constants.%Greater.6a2449.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_5.2 => constants.%Greater.6a2449.2
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.bba
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.442
// CHECK:STDOUT:   %.loc45_5.2 => constants.%.3a7
// CHECK:STDOUT:   %impl.elem3.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc40(constants.%AllFunctions, constants.%OrderedWith.facet.ce9) {
// CHECK:STDOUT:   %U.loc40_17.1 => constants.%AllFunctions
// CHECK:STDOUT:   %OrderedWith.type.loc40_48.1 => constants.%OrderedWith.type.549
// CHECK:STDOUT:   %T.loc40_27.1 => constants.%OrderedWith.facet.ce9
// CHECK:STDOUT:   %pattern_type.loc40_27 => constants.%pattern_type.4a9
// CHECK:STDOUT:   %T.as_type.loc40_54.1 => constants.%AllFunctions
// CHECK:STDOUT:   %pattern_type.loc40_52 => constants.%pattern_type.a19
// CHECK:STDOUT:   %pattern_type.loc40_58 => constants.%pattern_type.a19
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc40_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type.5d1
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.488
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a0f
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.26e
// CHECK:STDOUT:   %.loc42_5.2 => constants.%.7fa
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.63e
// CHECK:STDOUT:   %impl.elem0.loc42_5.2 => constants.%Less.8f0ac4.2
// CHECK:STDOUT:   %specific_impl_fn.loc42_5.2 => constants.%Less.8f0ac4.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d6f
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.c81
// CHECK:STDOUT:   %.loc43_5.2 => constants.%.6b8
// CHECK:STDOUT:   %impl.elem1.loc43_5.2 => constants.%LessOrEquivalent.9a8233.2
// CHECK:STDOUT:   %specific_impl_fn.loc43_5.2 => constants.%LessOrEquivalent.9a8233.2
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.2fa
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.898
// CHECK:STDOUT:   %.loc44_5.2 => constants.%.580
// CHECK:STDOUT:   %impl.elem2.loc44_5.2 => constants.%Greater.6a2449.3
// CHECK:STDOUT:   %specific_impl_fn.loc44_5.2 => constants.%Greater.6a2449.3
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.8e6
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.130
// CHECK:STDOUT:   %.loc45_5.2 => constants.%.cfa
// CHECK:STDOUT:   %impl.elem3.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.3
// CHECK:STDOUT:   %specific_impl_fn.loc45_5.2 => constants.%GreaterOrEquivalent.a3b59c.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ca8786.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.adc9f7.1: %OrderedWith.type.ca8786.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2b366e.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.975556.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6f28f1.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c82b75.1: %OrderedWith.WithSelf.Greater.type.6f28f1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.970504.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.86efc4.1: %OrderedWith.WithSelf.LessOrEquivalent.type.970504.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.df27e9.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.adc9f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.1a4163.1: %OrderedWith.WithSelf.Less.type.df27e9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.ca8786.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.d3f: type = pattern_type %OrderedWith.type.ca8786.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.ca8786.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2b366e.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.f6e965.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element0, imports.%Core.import_ref.255 [symbolic]
// CHECK:STDOUT:   %assoc1.23402c.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element1, imports.%Core.import_ref.857 [symbolic]
// CHECK:STDOUT:   %assoc2.771acd.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic]
// CHECK:STDOUT:   %assoc3.ba2aca.2: %OrderedWith.assoc_type.2b366e.2 = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.c70b34.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.3d2: <witness> = require_complete_type %OrderedWith.type.ca8786.2 [symbolic]
// CHECK:STDOUT:   %assoc0.666: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element0, imports.%Core.import_ref.e2b [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.df27e9.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.970504.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6f28f1.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.0af: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.df27e9.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.0af = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.072: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.e40: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element1, imports.%Core.import_ref.ce4 [symbolic]
// CHECK:STDOUT:   %.81b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.970504.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.81b = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f8: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.b0e: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element2, imports.%Core.import_ref.40e [symbolic]
// CHECK:STDOUT:   %.f69: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.6f28f1.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.f69 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1cd: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.078: %OrderedWith.assoc_type.2b366e.1 = assoc_entity element3, imports.%Core.import_ref.a50 [symbolic]
// CHECK:STDOUT:   %.206: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.206 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b2f: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.57d: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.c01: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %pattern_type.6ac: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %pattern_type.fe2: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %A3: type = class_type @A3 [concrete]
// CHECK:STDOUT:   %pattern_type.bb4: type = pattern_type %A3 [concrete]
// CHECK:STDOUT:   %B3: type = class_type @B3 [concrete]
// CHECK:STDOUT:   %pattern_type.b85: type = pattern_type %B3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.773: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.5e8: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.d03: %OrderedWith.assoc_type.5e8 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.f6b: %OrderedWith.assoc_type.5e8 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.911: %OrderedWith.assoc_type.5e8 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.a71: %OrderedWith.assoc_type.5e8 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e39: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.0fa: %OrderedWith.type.773 = facet_value %A1, (%custom_witness.e39) [concrete]
// CHECK:STDOUT:   %pattern_type.f4b: type = pattern_type %OrderedWith.type.773 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.cf0: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.42c: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.4b4: %OrderedWith.assoc_type.42c = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.e14: %OrderedWith.assoc_type.42c = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.733: %OrderedWith.assoc_type.42c = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.ffa: %OrderedWith.assoc_type.42c = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b81: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.d5c: %OrderedWith.type.cf0 = facet_value %B1, (%custom_witness.b81) [concrete]
// CHECK:STDOUT:   %pattern_type.3f7: type = pattern_type %OrderedWith.type.cf0 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.26a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.688: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.a70: %OrderedWith.assoc_type.688 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.54d: %OrderedWith.assoc_type.688 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.15d: %OrderedWith.assoc_type.688 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.68c: %OrderedWith.assoc_type.688 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.176: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.edc: %OrderedWith.type.26a = facet_value %B2, (%custom_witness.176) [concrete]
// CHECK:STDOUT:   %pattern_type.899: type = pattern_type %OrderedWith.type.26a [concrete]
// CHECK:STDOUT:   %OrderedWith.type.232: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.02d: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.0d8: %OrderedWith.assoc_type.02d = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.f4d: %OrderedWith.assoc_type.02d = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.ba7: %OrderedWith.assoc_type.02d = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.fa1: %OrderedWith.assoc_type.02d = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.499: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.ba3: %OrderedWith.type.232 = facet_value %A3, (%custom_witness.499) [concrete]
// CHECK:STDOUT:   %pattern_type.f06: type = pattern_type %OrderedWith.type.232 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.343: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.8f8: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.4c0: %OrderedWith.assoc_type.8f8 = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.c91: %OrderedWith.assoc_type.8f8 = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.7b7: %OrderedWith.assoc_type.8f8 = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.1c2: %OrderedWith.assoc_type.8f8 = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.679: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.b33: %OrderedWith.type.343 = facet_value %B3, (%custom_witness.679) [concrete]
// CHECK:STDOUT:   %pattern_type.e5d: type = pattern_type %OrderedWith.type.343 [concrete]
// CHECK:STDOUT:   %complete_type.68a: <witness> = complete_type_witness %OrderedWith.type.773 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.e0f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.0fa) [concrete]
// CHECK:STDOUT:   %.e26: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.e0f, %OrderedWith.facet.0fa [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.15d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.0fa) [concrete]
// CHECK:STDOUT:   %.c48: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.15d, %OrderedWith.facet.0fa [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.3a1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.0fa) [concrete]
// CHECK:STDOUT:   %.dba: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.3a1, %OrderedWith.facet.0fa [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ccd: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.0fa) [concrete]
// CHECK:STDOUT:   %.64a: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ccd, %OrderedWith.facet.0fa [concrete]
// CHECK:STDOUT:   %complete_type.ee8: <witness> = complete_type_witness %OrderedWith.type.cf0 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f2f: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.d5c) [concrete]
// CHECK:STDOUT:   %.d99: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f2f, %OrderedWith.facet.d5c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.64e: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.d5c) [concrete]
// CHECK:STDOUT:   %.d5b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.64e, %OrderedWith.facet.d5c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.4af: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.d5c) [concrete]
// CHECK:STDOUT:   %.b8a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.4af, %OrderedWith.facet.d5c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.037: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.d5c) [concrete]
// CHECK:STDOUT:   %.edc: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.037, %OrderedWith.facet.d5c [concrete]
// CHECK:STDOUT:   %complete_type.518: <witness> = complete_type_witness %OrderedWith.type.26a [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c1b: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.edc) [concrete]
// CHECK:STDOUT:   %.ff0: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c1b, %OrderedWith.facet.edc [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.341: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.edc) [concrete]
// CHECK:STDOUT:   %.d0f: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.341, %OrderedWith.facet.edc [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.bb0: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.edc) [concrete]
// CHECK:STDOUT:   %.e4f: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.bb0, %OrderedWith.facet.edc [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.c69: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.edc) [concrete]
// CHECK:STDOUT:   %.fe6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.c69, %OrderedWith.facet.edc [concrete]
// CHECK:STDOUT:   %complete_type.7ba: <witness> = complete_type_witness %OrderedWith.type.232 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.3f8: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.ba3) [concrete]
// CHECK:STDOUT:   %.7f9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.3f8, %OrderedWith.facet.ba3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.a19: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.ba3) [concrete]
// CHECK:STDOUT:   %.d93: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.a19, %OrderedWith.facet.ba3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ae5: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.ba3) [concrete]
// CHECK:STDOUT:   %.152: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ae5, %OrderedWith.facet.ba3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.17e: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.ba3) [concrete]
// CHECK:STDOUT:   %.4d1: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.17e, %OrderedWith.facet.ba3 [concrete]
// CHECK:STDOUT:   %complete_type.af4: <witness> = complete_type_witness %OrderedWith.type.343 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.444: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.b33) [concrete]
// CHECK:STDOUT:   %.5ac: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.444, %OrderedWith.facet.b33 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.e9d: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.b33) [concrete]
// CHECK:STDOUT:   %.158: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.e9d, %OrderedWith.facet.b33 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.7e9: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.b33) [concrete]
// CHECK:STDOUT:   %.8ef: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.7e9, %OrderedWith.facet.b33 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.408: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.b33) [concrete]
// CHECK:STDOUT:   %.002: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.408, %OrderedWith.facet.b33 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e9: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.666)]
// CHECK:STDOUT:   %Core.import_ref.c86: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.e40)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.b0e)]
// CHECK:STDOUT:   %Core.import_ref.065: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.078)]
// CHECK:STDOUT:   %Core.import_ref.bb2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.975556.1)]
// CHECK:STDOUT:   %Core.import_ref.a0a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.6f28f1.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c82b75.1)]
// CHECK:STDOUT:   %Core.import_ref.857: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.970504.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.86efc4.1)]
// CHECK:STDOUT:   %Core.import_ref.255: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.df27e9.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.1a4163.1)]
// CHECK:STDOUT:   %Core.import_ref.e2b = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ce4 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.40e = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a50 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc63(%U.loc63_17.2: type, %T.loc63_27.2: @OrderedWith.loc63.%OrderedWith.type.loc63_48.1 (%OrderedWith.type.ca8786.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc65: <witness> = require_complete_type %OrderedWith.type.loc63_48.1 [symbolic = %require_complete.loc65 (constants.%require_complete.3d2)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc63_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2b366e.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element0, imports.%Core.import_ref.255 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.df27e9.3)]
// CHECK:STDOUT:   %.loc65_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc63_27.1 [symbolic = %.loc65_5.2 (constants.%.0af)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc63_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc63_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc65_5.2: @OrderedWith.loc63.%.loc65_5.2 (%.0af) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc65_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2: <specific function> = specific_impl_function %impl.elem0.loc65_5.2, @OrderedWith.WithSelf.Less(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %specific_impl_fn.loc65_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element1, imports.%Core.import_ref.857 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.970504.3)]
// CHECK:STDOUT:   %.loc66_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc63_27.1 [symbolic = %.loc66_5.2 (constants.%.81b)]
// CHECK:STDOUT:   %impl.elem1.loc66_5.2: @OrderedWith.loc63.%.loc66_5.2 (%.81b) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc66_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2: <specific function> = specific_impl_function %impl.elem1.loc66_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %specific_impl_fn.loc66_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.6f28f1.3)]
// CHECK:STDOUT:   %.loc67_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc63_27.1 [symbolic = %.loc67_5.2 (constants.%.f69)]
// CHECK:STDOUT:   %impl.elem2.loc67_5.2: @OrderedWith.loc63.%.loc67_5.2 (%.f69) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc67_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2: <specific function> = specific_impl_function %impl.elem2.loc67_5.2, @OrderedWith.WithSelf.Greater(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %specific_impl_fn.loc67_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3)]
// CHECK:STDOUT:   %.loc68_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc63_27.1 [symbolic = %.loc68_5.2 (constants.%.206)]
// CHECK:STDOUT:   %impl.elem3.loc68_5.2: @OrderedWith.loc63.%.loc68_5.2 (%.206) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc68_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2: <specific function> = specific_impl_function %impl.elem3.loc68_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc63_17.1, %T.loc63_27.1) [symbolic = %specific_impl_fn.loc68_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc63.%T.as_type.loc63_54.1 (%T.as_type), %y.param: @OrderedWith.loc63.%U.loc63_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc65: @OrderedWith.loc63.%T.as_type.loc63_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc65: @OrderedWith.loc63.%U.loc63_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc65: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc63_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc65_5.1: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.7e9, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Less, %.loc65_5.1 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %impl.elem0.loc65_5.1: @OrderedWith.loc63.%.loc65_5.2 (%.0af) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc65_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc65_5.1: <bound method> = bound_method %x.ref.loc65, %impl.elem0.loc65_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc65_5.1: <specific function> = specific_impl_function %impl.elem0.loc65_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc65_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:     %bound_method.loc65_5.2: <bound method> = bound_method %x.ref.loc65, %specific_impl_fn.loc65_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc65_5.2(%x.ref.loc65, %y.ref.loc65)
// CHECK:STDOUT:     %x.ref.loc66: @OrderedWith.loc63.%T.as_type.loc63_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc66: @OrderedWith.loc63.%U.loc63_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc66: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc63_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc66_5.1: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.c86, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref LessOrEquivalent, %.loc66_5.1 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %impl.elem1.loc66_5.1: @OrderedWith.loc63.%.loc66_5.2 (%.81b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc66_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc66_5.1: <bound method> = bound_method %x.ref.loc66, %impl.elem1.loc66_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc66_5.1: <specific function> = specific_impl_function %impl.elem1.loc66_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc66_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc66_5.2: <bound method> = bound_method %x.ref.loc66, %specific_impl_fn.loc66_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc66_5.2(%x.ref.loc66, %y.ref.loc66)
// CHECK:STDOUT:     %x.ref.loc67: @OrderedWith.loc63.%T.as_type.loc63_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc67: @OrderedWith.loc63.%U.loc63_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc67: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc63_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc67_5.1: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.f4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Greater, %.loc67_5.1 [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %impl.elem2.loc67_5.1: @OrderedWith.loc63.%.loc67_5.2 (%.f69) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc67_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc67_5.1: <bound method> = bound_method %x.ref.loc67, %impl.elem2.loc67_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc67_5.1: <specific function> = specific_impl_function %impl.elem2.loc67_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc67_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:     %bound_method.loc67_5.2: <bound method> = bound_method %x.ref.loc67, %specific_impl_fn.loc67_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc67_5.2(%x.ref.loc67, %y.ref.loc67)
// CHECK:STDOUT:     %x.ref.loc68: @OrderedWith.loc63.%T.as_type.loc63_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc68: @OrderedWith.loc63.%U.loc63_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc68: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc63_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc68_5.1: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.065, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc63.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref GreaterOrEquivalent, %.loc68_5.1 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %impl.elem3.loc68_5.1: @OrderedWith.loc63.%.loc68_5.2 (%.206) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc68_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc68_5.1: <bound method> = bound_method %x.ref.loc68, %impl.elem3.loc68_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc68_5.1: <specific function> = specific_impl_function %impl.elem3.loc68_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc68_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:     %bound_method.loc68_5.2: <bound method> = bound_method %x.ref.loc68, %specific_impl_fn.loc68_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc68_5.2(%x.ref.loc68, %y.ref.loc68)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.ca8786.2
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.d3f
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.c70b34.2
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%B1, constants.%OrderedWith.facet.0fa) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.773
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%OrderedWith.facet.0fa
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.f4b
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.57d
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.c01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc63_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc63_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.68a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.5e8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d03
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.e0f
// CHECK:STDOUT:   %.loc65_5.2 => constants.%.e26
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.e39
// CHECK:STDOUT:   %impl.elem0.loc65_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f6b
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.15d
// CHECK:STDOUT:   %.loc66_5.2 => constants.%.c48
// CHECK:STDOUT:   %impl.elem1.loc66_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.911
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.3a1
// CHECK:STDOUT:   %.loc67_5.2 => constants.%.dba
// CHECK:STDOUT:   %impl.elem2.loc67_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.a71
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.ccd
// CHECK:STDOUT:   %.loc68_5.2 => constants.%.64a
// CHECK:STDOUT:   %impl.elem3.loc68_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%A1, constants.%OrderedWith.facet.d5c) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.cf0
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%OrderedWith.facet.d5c
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.3f7
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.c01
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.57d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc63_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc63_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.ee8
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.42c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4b4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.f2f
// CHECK:STDOUT:   %.loc65_5.2 => constants.%.d99
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.b81
// CHECK:STDOUT:   %impl.elem0.loc65_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e14
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.64e
// CHECK:STDOUT:   %.loc66_5.2 => constants.%.d5b
// CHECK:STDOUT:   %impl.elem1.loc66_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.733
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.4af
// CHECK:STDOUT:   %.loc67_5.2 => constants.%.b8a
// CHECK:STDOUT:   %impl.elem2.loc67_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.ffa
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.037
// CHECK:STDOUT:   %.loc68_5.2 => constants.%.edc
// CHECK:STDOUT:   %impl.elem3.loc68_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%A2, constants.%OrderedWith.facet.edc) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.26a
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%OrderedWith.facet.edc
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.899
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.fe2
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.6ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc63_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc63_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.518
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.688
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a70
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.c1b
// CHECK:STDOUT:   %.loc65_5.2 => constants.%.ff0
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.176
// CHECK:STDOUT:   %impl.elem0.loc65_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.54d
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.341
// CHECK:STDOUT:   %.loc66_5.2 => constants.%.d0f
// CHECK:STDOUT:   %impl.elem1.loc66_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.15d
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.bb0
// CHECK:STDOUT:   %.loc67_5.2 => constants.%.e4f
// CHECK:STDOUT:   %impl.elem2.loc67_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.68c
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.c69
// CHECK:STDOUT:   %.loc68_5.2 => constants.%.fe6
// CHECK:STDOUT:   %impl.elem3.loc68_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%B3, constants.%OrderedWith.facet.ba3) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.232
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%OrderedWith.facet.ba3
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.f06
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.bb4
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.b85
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc63_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc63_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.7ba
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.02d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0d8
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.3f8
// CHECK:STDOUT:   %.loc65_5.2 => constants.%.7f9
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.499
// CHECK:STDOUT:   %impl.elem0.loc65_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f4d
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.a19
// CHECK:STDOUT:   %.loc66_5.2 => constants.%.d93
// CHECK:STDOUT:   %impl.elem1.loc66_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.ba7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.ae5
// CHECK:STDOUT:   %.loc67_5.2 => constants.%.152
// CHECK:STDOUT:   %impl.elem2.loc67_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.fa1
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.17e
// CHECK:STDOUT:   %.loc68_5.2 => constants.%.4d1
// CHECK:STDOUT:   %impl.elem3.loc68_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc63(constants.%A3, constants.%OrderedWith.facet.b33) {
// CHECK:STDOUT:   %U.loc63_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc63_48.1 => constants.%OrderedWith.type.343
// CHECK:STDOUT:   %T.loc63_27.1 => constants.%OrderedWith.facet.b33
// CHECK:STDOUT:   %pattern_type.loc63_27 => constants.%pattern_type.e5d
// CHECK:STDOUT:   %T.as_type.loc63_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc63_52 => constants.%pattern_type.b85
// CHECK:STDOUT:   %pattern_type.loc63_58 => constants.%pattern_type.bb4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc63_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc63_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc65 => constants.%complete_type.af4
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.8f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4c0
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.444
// CHECK:STDOUT:   %.loc65_5.2 => constants.%.5ac
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.679
// CHECK:STDOUT:   %impl.elem0.loc65_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc65_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c91
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.e9d
// CHECK:STDOUT:   %.loc66_5.2 => constants.%.158
// CHECK:STDOUT:   %impl.elem1.loc66_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc66_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.7b7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.7e9
// CHECK:STDOUT:   %.loc67_5.2 => constants.%.8ef
// CHECK:STDOUT:   %impl.elem2.loc67_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc67_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.1c2
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.408
// CHECK:STDOUT:   %.loc68_5.2 => constants.%.002
// CHECK:STDOUT:   %impl.elem3.loc68_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc68_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
