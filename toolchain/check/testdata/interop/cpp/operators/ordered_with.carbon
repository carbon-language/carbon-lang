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

// --- returns_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct Ordered {
  auto operator<(Ordered const&) const -> bool;
  auto operator<=(Ordered const&) const -> bool;
  auto operator>(Ordered const&) const -> bool;
  auto operator>=(Ordered const&) const -> bool;
};
''';

fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x < y;
  x <= y;
  x > y;
  x >= y;
  //@dump-sem-ir-end
}

fn Test(ordered: Cpp.Ordered) {
  OrderedWith(ordered, ordered);
}

// --- cross_type_ordering.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
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

struct B2;
struct A2 {};
struct B2 {
  auto operator<(A2) const -> bool;
  auto operator<=(A2) const -> bool;
  auto operator>(A2) const -> bool;
  auto operator>=(A2) const -> bool;
};

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
struct LessReturnsBoolish {
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+9]]:8: error: cannot implicitly convert expression of type `i32` to `bool` [ConversionFailure]
  // CHECK:STDERR:   auto operator<(LessReturnsBoolish) const -> int;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+6]]:8: note: type `i32` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   auto operator<(LessReturnsBoolish) const -> int;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:26:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Less[self: Self](other: Other) -> bool;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  auto operator<(LessReturnsBoolish) const -> int;
  auto operator<=(LessReturnsBoolish) const -> bool;
  auto operator>(LessReturnsBoolish) const -> bool;
  auto operator>=(LessReturnsBoolish) const -> bool;
};

struct LessEqualReturnsBoolish {
  auto operator<(LessEqualReturnsBoolish) const -> bool;
  auto operator<=(LessEqualReturnsBoolish) const -> int*;
  auto operator>(LessEqualReturnsBoolish) const -> bool;
  auto operator>=(LessEqualReturnsBoolish) const -> bool;
};

struct GreaterEqualReturnsBoolish {
  enum Result { False, True };
  auto operator<(GreaterEqualReturnsBoolish) const -> bool;
  auto operator<=(GreaterEqualReturnsBoolish) const -> bool;
  auto operator>(GreaterEqualReturnsBoolish) const -> bool;
  auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
};

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

struct AllReturnBoolish {
  auto operator<(AllReturnBoolish) const -> double;
  auto operator<=(AllReturnBoolish) const -> double;
  auto operator>(AllReturnBoolish) const -> double;
  auto operator>=(AllReturnBoolish) const -> double;
};
''';

// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+82]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-36]]:8: error: cannot implicitly convert expression of type `Core.Optional(i32* as Core.OptionalStorage)` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<=(LessEqualReturnsBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-39]]:8: note: type `Core.Optional(i32* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<=(LessEqualReturnsBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:27:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn LessOrEquivalent[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+69]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-39]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-42]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:29:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn GreaterOrEquivalent[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+56]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-37]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-40]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:26:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Less[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-49]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-52]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:27:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn LessOrEquivalent[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+30]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-61]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-64]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:28:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Greater[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-73]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-76]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:29:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn GreaterOrEquivalent[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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
  OrderedWith(less_returns_boolish, less_returns_boolish);

  OrderedWith(less_equal_returns_boolish, less_equal_returns_boolish);

  OrderedWith(greater_equal_returns_boolish, greater_equal_returns_boolish);

  OrderedWith(greater_returns_boolish, greater_returns_boolish);

  OrderedWith(all_return_boolish, all_return_boolish);
}

// --- fail_never_valid.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct MissingLess {
  auto operator<=(MissingLess) const -> bool;
  auto operator>(MissingLess) const -> bool;
  auto operator>=(MissingLess) const -> bool;
};

struct MissingLessEqual {
  auto operator<(MissingLessEqual) const -> bool;
  auto operator>(MissingLessEqual) const -> bool;
  auto operator>=(MissingLessEqual) const -> bool;
};

struct MissingGreater {
  auto operator<(MissingGreater) const -> bool;
  auto operator<=(MissingGreater) const -> bool;
  auto operator>=(MissingGreater) const -> bool;
};

struct MissingGreaterEqual {
  auto operator<(MissingGreaterEqual) const -> bool;
  auto operator<=(MissingGreaterEqual) const -> bool;
  auto operator>(MissingGreaterEqual) const -> bool;
};

struct NoRelevantMembers {};

struct LessReturnsVoid {
  auto operator<(LessReturnsVoid) const -> void;
  auto operator<=(LessReturnsVoid) const -> bool;
  auto operator>(LessReturnsVoid) const -> bool;
  auto operator>=(LessReturnsVoid) const -> bool;
};

struct LessNonConst {
  auto operator<(LessNonConst) -> bool;
  auto operator<=(LessNonConst) const -> bool;
  auto operator>(LessNonConst) const -> bool;
  auto operator>=(LessNonConst) const -> bool;
};
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
        less_returns_void: Cpp.LessReturnsVoid,
        less_non_const: Cpp.LessNonConst) {
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingLess` into type implementing `Core.OrderedWith(Cpp.MissingLess)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_less, missing_less);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less, missing_less);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingLessEqual` into type implementing `Core.OrderedWith(Cpp.MissingLessEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_less_equal, missing_less_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-26]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less_equal, missing_less_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreater` into type implementing `Core.OrderedWith(Cpp.MissingGreater)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater, missing_greater);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater, missing_greater);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreaterEqual` into type implementing `Core.OrderedWith(Cpp.MissingGreaterEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater_equal, missing_greater_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-44]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater_equal, missing_greater_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoRelevantMembers` into type implementing `Core.OrderedWith(Cpp.NoRelevantMembers)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(no_relevant_members, no_relevant_members);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-53]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(no_relevant_members, no_relevant_members);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.LessReturnsVoid` into type implementing `Core.OrderedWith(Cpp.LessReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(less_returns_void, less_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-62]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_returns_void, less_returns_void);

  OrderedWith(less_non_const, less_non_const);
}

// CHECK:STDOUT: --- returns_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.33ce9e.1: %OrderedWith.type.c171ee.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c73b41.1: %OrderedWith.WithSelf.Greater.type.c74212.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.1dd275.1: %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.cefdc0.1: %OrderedWith.WithSelf.Less.type.123104.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.e98: type = pattern_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.c171ee.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.a764f9.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic]
// CHECK:STDOUT:   %assoc1.05b62d.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic]
// CHECK:STDOUT:   %assoc2.19f303.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic]
// CHECK:STDOUT:   %assoc3.3683e0.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element3, imports.%Core.import_ref.095 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.0cd5d4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.600: <witness> = require_complete_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %assoc0.e6a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element0, imports.%Core.import_ref.52f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.c4c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.123104.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.c4c = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e3: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.224: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element1, imports.%Core.import_ref.f4c [symbolic]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.3d5 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.27d: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.2dc: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element2, imports.%Core.import_ref.f2c [symbolic]
// CHECK:STDOUT:   %.7e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.c74212.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.7e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b70: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.34a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element3, imports.%Core.import_ref.209 [symbolic]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.71b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f8: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.6d1: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.0d8: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.511: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.087: %OrderedWith.assoc_type.511 = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.a7c: %OrderedWith.assoc_type.511 = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.7ad: %OrderedWith.assoc_type.511 = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.1fd: %OrderedWith.assoc_type.511 = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.462: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.6ac: %OrderedWith.type.0d8 = facet_value %Ordered, (%custom_witness.462) [concrete]
// CHECK:STDOUT:   %pattern_type.426: type = pattern_type %OrderedWith.type.0d8 [concrete]
// CHECK:STDOUT:   %complete_type.f9c: <witness> = complete_type_witness %OrderedWith.type.0d8 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.5bb: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.6ac) [concrete]
// CHECK:STDOUT:   %.4b2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.5bb, %OrderedWith.facet.6ac [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.439: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.6ac) [concrete]
// CHECK:STDOUT:   %.00e: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.439, %OrderedWith.facet.6ac [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.ea2: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.6ac) [concrete]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.ea2, %OrderedWith.facet.6ac [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.aaa: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.6ac) [concrete]
// CHECK:STDOUT:   %.d8b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.aaa, %OrderedWith.facet.6ac [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.246: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.e6a)]
// CHECK:STDOUT:   %Core.import_ref.8a6: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.224)]
// CHECK:STDOUT:   %Core.import_ref.f62: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.2dc)]
// CHECK:STDOUT:   %Core.import_ref.9ce: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.34a)]
// CHECK:STDOUT:   %Core.import_ref.095: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1)]
// CHECK:STDOUT:   %Core.import_ref.f8b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.c74212.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c73b41.1)]
// CHECK:STDOUT:   %Core.import_ref.6b8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.1dd275.1)]
// CHECK:STDOUT:   %Core.import_ref.9d2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.123104.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.cefdc0.1)]
// CHECK:STDOUT:   %Core.import_ref.52f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f4c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f2c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.209 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_27.2: @OrderedWith.loc13.%OrderedWith.type.loc13_48.1 (%OrderedWith.type.c171ee.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_48.1 [symbolic = %require_complete.loc15 (constants.%require_complete.600)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2eba71.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.123104.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_27.1 [symbolic = %.loc15_5.2 (constants.%.c4c)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.c4c) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc16_5.2 (constants.%.3d5)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.3d5) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.c74212.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_27.1 [symbolic = %.loc17_5.2 (constants.%.7e7)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.7e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element3, imports.%Core.import_ref.095 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc18_5.2 (constants.%.71b)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.71b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.246, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.c4c) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.8a6, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.3d5) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.f62, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.7e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.9ce, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.71b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.c171ee.2
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.e98
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.0cd5d4.2
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.6ac) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.0d8
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%OrderedWith.facet.6ac
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.426
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.6d1
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.6d1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.f9c
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.511
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.087
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.5bb
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.4b2
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.462
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a7c
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.439
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.00e
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.7ad
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.ea2
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.72e
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.1fd
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.aaa
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.d8b
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.33ce9e.1: %OrderedWith.type.c171ee.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c73b41.1: %OrderedWith.WithSelf.Greater.type.c74212.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.1dd275.1: %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.33ce9e.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.cefdc0.1: %OrderedWith.WithSelf.Less.type.123104.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.c171ee.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.e98: type = pattern_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.c171ee.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2eba71.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.a764f9.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic]
// CHECK:STDOUT:   %assoc1.05b62d.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic]
// CHECK:STDOUT:   %assoc2.19f303.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic]
// CHECK:STDOUT:   %assoc3.3683e0.2: %OrderedWith.assoc_type.2eba71.2 = assoc_entity element3, imports.%Core.import_ref.095 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.0cd5d4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.600: <witness> = require_complete_type %OrderedWith.type.c171ee.2 [symbolic]
// CHECK:STDOUT:   %assoc0.e6a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element0, imports.%Core.import_ref.52f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.123104.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.c74212.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.c4c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.123104.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.c4c = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5e3: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.224: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element1, imports.%Core.import_ref.f4c [symbolic]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.3d5 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.27d: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.2dc: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element2, imports.%Core.import_ref.f2c [symbolic]
// CHECK:STDOUT:   %.7e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.c74212.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.7e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b70: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.34a: %OrderedWith.assoc_type.2eba71.1 = assoc_entity element3, imports.%Core.import_ref.209 [symbolic]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.71b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f8: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.c94: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.d4a: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %pattern_type.3d2: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %pattern_type.c79: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %A3: type = class_type @A3 [concrete]
// CHECK:STDOUT:   %pattern_type.8e6: type = pattern_type %A3 [concrete]
// CHECK:STDOUT:   %B3: type = class_type @B3 [concrete]
// CHECK:STDOUT:   %pattern_type.4e7: type = pattern_type %B3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.652: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1b6: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.c19: %OrderedWith.assoc_type.1b6 = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.82c: %OrderedWith.assoc_type.1b6 = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.3bf: %OrderedWith.assoc_type.1b6 = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.a1b: %OrderedWith.assoc_type.1b6 = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d09: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.775: %OrderedWith.type.652 = facet_value %A1, (%custom_witness.d09) [concrete]
// CHECK:STDOUT:   %pattern_type.b33: type = pattern_type %OrderedWith.type.652 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.2ae: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.769: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.25f: %OrderedWith.assoc_type.769 = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.ae7: %OrderedWith.assoc_type.769 = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.e15: %OrderedWith.assoc_type.769 = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.595: %OrderedWith.assoc_type.769 = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.63e: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.e1c: %OrderedWith.type.2ae = facet_value %B1, (%custom_witness.63e) [concrete]
// CHECK:STDOUT:   %pattern_type.716: type = pattern_type %OrderedWith.type.2ae [concrete]
// CHECK:STDOUT:   %OrderedWith.type.777: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.c7e: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.f52: %OrderedWith.assoc_type.c7e = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.ff0: %OrderedWith.assoc_type.c7e = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.8dc: %OrderedWith.assoc_type.c7e = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.349: %OrderedWith.assoc_type.c7e = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.04d: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.81f: %OrderedWith.type.777 = facet_value %B2, (%custom_witness.04d) [concrete]
// CHECK:STDOUT:   %pattern_type.357: type = pattern_type %OrderedWith.type.777 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9d5: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.f58: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.26d: %OrderedWith.assoc_type.f58 = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.0b2: %OrderedWith.assoc_type.f58 = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.a9f: %OrderedWith.assoc_type.f58 = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.dc9: %OrderedWith.assoc_type.f58 = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cfa: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.e8b: %OrderedWith.type.9d5 = facet_value %A3, (%custom_witness.cfa) [concrete]
// CHECK:STDOUT:   %pattern_type.06c: type = pattern_type %OrderedWith.type.9d5 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.d8d: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e0b: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.616: %OrderedWith.assoc_type.e0b = assoc_entity element0, imports.%Core.import_ref.9d2 [concrete]
// CHECK:STDOUT:   %assoc1.469: %OrderedWith.assoc_type.e0b = assoc_entity element1, imports.%Core.import_ref.6b8 [concrete]
// CHECK:STDOUT:   %assoc2.7fd: %OrderedWith.assoc_type.e0b = assoc_entity element2, imports.%Core.import_ref.f8b [concrete]
// CHECK:STDOUT:   %assoc3.89c: %OrderedWith.assoc_type.e0b = assoc_entity element3, imports.%Core.import_ref.095 [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7e9: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fe5: %OrderedWith.type.d8d = facet_value %B3, (%custom_witness.7e9) [concrete]
// CHECK:STDOUT:   %pattern_type.975: type = pattern_type %OrderedWith.type.d8d [concrete]
// CHECK:STDOUT:   %complete_type.85d: <witness> = complete_type_witness %OrderedWith.type.652 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.523: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.775) [concrete]
// CHECK:STDOUT:   %.e6f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.523, %OrderedWith.facet.775 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.608: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.775) [concrete]
// CHECK:STDOUT:   %.b05: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.608, %OrderedWith.facet.775 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.83b: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.775) [concrete]
// CHECK:STDOUT:   %.9ee: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.83b, %OrderedWith.facet.775 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.8af: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.775) [concrete]
// CHECK:STDOUT:   %.cb9: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.8af, %OrderedWith.facet.775 [concrete]
// CHECK:STDOUT:   %complete_type.f3a: <witness> = complete_type_witness %OrderedWith.type.2ae [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f38: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.e1c) [concrete]
// CHECK:STDOUT:   %.90a: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f38, %OrderedWith.facet.e1c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.42e: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.e1c) [concrete]
// CHECK:STDOUT:   %.e39: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.42e, %OrderedWith.facet.e1c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.10a: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.e1c) [concrete]
// CHECK:STDOUT:   %.8f9: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.10a, %OrderedWith.facet.e1c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.039: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.e1c) [concrete]
// CHECK:STDOUT:   %.36e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.039, %OrderedWith.facet.e1c [concrete]
// CHECK:STDOUT:   %complete_type.77a: <witness> = complete_type_witness %OrderedWith.type.777 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.efa: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.81f) [concrete]
// CHECK:STDOUT:   %.746: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.efa, %OrderedWith.facet.81f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.0d7: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.81f) [concrete]
// CHECK:STDOUT:   %.3c2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.0d7, %OrderedWith.facet.81f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d9c: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.81f) [concrete]
// CHECK:STDOUT:   %.00f: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d9c, %OrderedWith.facet.81f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.854: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.81f) [concrete]
// CHECK:STDOUT:   %.e37: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.854, %OrderedWith.facet.81f [concrete]
// CHECK:STDOUT:   %complete_type.dca: <witness> = complete_type_witness %OrderedWith.type.9d5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.f09: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.e8b) [concrete]
// CHECK:STDOUT:   %.029: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.f09, %OrderedWith.facet.e8b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.d96: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.e8b) [concrete]
// CHECK:STDOUT:   %.30a: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.d96, %OrderedWith.facet.e8b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6cb: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.e8b) [concrete]
// CHECK:STDOUT:   %.dc8: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.6cb, %OrderedWith.facet.e8b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.161: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.e8b) [concrete]
// CHECK:STDOUT:   %.66c: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.161, %OrderedWith.facet.e8b [concrete]
// CHECK:STDOUT:   %complete_type.771: <witness> = complete_type_witness %OrderedWith.type.d8d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.e00: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.fe5) [concrete]
// CHECK:STDOUT:   %.047: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.e00, %OrderedWith.facet.fe5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.6bf: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.fe5) [concrete]
// CHECK:STDOUT:   %.1ea: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.6bf, %OrderedWith.facet.fe5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9b8: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.fe5) [concrete]
// CHECK:STDOUT:   %.850: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9b8, %OrderedWith.facet.fe5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d44: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.fe5) [concrete]
// CHECK:STDOUT:   %.cd6: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d44, %OrderedWith.facet.fe5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.246: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.e6a)]
// CHECK:STDOUT:   %Core.import_ref.8a6: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.224)]
// CHECK:STDOUT:   %Core.import_ref.f62: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.2dc)]
// CHECK:STDOUT:   %Core.import_ref.9ce: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.34a)]
// CHECK:STDOUT:   %Core.import_ref.095: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.4b6b94.1)]
// CHECK:STDOUT:   %Core.import_ref.f8b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.c74212.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c73b41.1)]
// CHECK:STDOUT:   %Core.import_ref.6b8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.1dd275.1)]
// CHECK:STDOUT:   %Core.import_ref.9d2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.123104.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.cefdc0.1)]
// CHECK:STDOUT:   %Core.import_ref.52f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f4c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f2c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.209 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_27.2: @OrderedWith.loc54.%OrderedWith.type.loc54_48.1 (%OrderedWith.type.c171ee.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_48.1 [symbolic = %require_complete.loc56 (constants.%require_complete.600)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2eba71.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element0, imports.%Core.import_ref.9d2 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.123104.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_27.1 [symbolic = %.loc56_5.2 (constants.%.c4c)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.c4c) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element1, imports.%Core.import_ref.6b8 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.3d0a17.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc57_5.2 (constants.%.3d5)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.3d5) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element2, imports.%Core.import_ref.f8b [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.c74212.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_27.1 [symbolic = %.loc58_5.2 (constants.%.7e7)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.7e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = assoc_entity element3, imports.%Core.import_ref.095 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a8b50e.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc59_5.2 (constants.%.71b)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.71b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.246, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.a764f9.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.c4c) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.5e3)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.8a6, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.05b62d.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.3d5) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.27d)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.f62, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.19f303.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.7e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.c171ee.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = specific_constant imports.%Core.import_ref.9ce, @OrderedWith.WithSelf(constants.%U, constants.%Self.33ce9e.1) [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2eba71.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.3683e0.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.71b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.c171ee.2
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.e98
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.0cd5d4.2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.775) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.652
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.775
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.b33
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c94
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.d4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.85d
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.1b6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c19
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.523
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.e6f
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.d09
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.82c
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.608
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.b05
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.3bf
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.83b
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.9ee
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.a1b
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.8af
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.cb9
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.e1c) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.2ae
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.e1c
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.716
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.d4a
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.c94
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.f3a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.769
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.25f
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.f38
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.90a
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.63e
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ae7
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.42e
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.e39
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.e15
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.10a
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.8f9
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.595
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.039
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.36e
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.81f) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.777
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.81f
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.357
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c79
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.3d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.77a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.c7e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f52
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.efa
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.746
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.04d
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ff0
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.0d7
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.3c2
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.8dc
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.d9c
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.00f
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.349
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.854
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.e37
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.e8b) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.9d5
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.e8b
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.06c
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.8e6
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.4e7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.dca
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.f58
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.26d
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.f09
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.029
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.cfa
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0b2
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.d96
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.30a
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.a9f
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.6cb
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.dc8
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.dc9
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.161
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.66c
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.fe5) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.d8d
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.fe5
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.975
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.4e7
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.8e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.771
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.e0b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.616
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.e00
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.047
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.7e9
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.469
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.6bf
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.1ea
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.7fd
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.9b8
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.850
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.89c
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.d44
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.cd6
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
