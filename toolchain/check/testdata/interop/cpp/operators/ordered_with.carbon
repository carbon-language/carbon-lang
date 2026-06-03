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
// CHECK:STDOUT:   %OrderedWith.type.73dab8.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d014af.1: %OrderedWith.type.73dab8.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.984598.1: %OrderedWith.WithSelf.Greater.type.9defd1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1: %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.4a995a.1: %OrderedWith.WithSelf.Less.type.687390.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.73dab8.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.10d: type = pattern_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.73dab8.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.7dbd55.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic]
// CHECK:STDOUT:   %assoc1.54de85.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic]
// CHECK:STDOUT:   %assoc2.3caaa9.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element2, imports.%Core.import_ref.32f [symbolic]
// CHECK:STDOUT:   %assoc3.4eb879.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.05039a.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.a45: <witness> = require_complete_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3ff: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element0, imports.%Core.import_ref.c33 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.687390.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.cb4 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c89: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.760: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element1, imports.%Core.import_ref.79a [symbolic]
// CHECK:STDOUT:   %.ac4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.ac4 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65b: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.897: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element2, imports.%Core.import_ref.ea2 [symbolic]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9defd1.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.994 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6dc: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.8bc: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element3, imports.%Core.import_ref.b4f [symbolic]
// CHECK:STDOUT:   %.eca: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.eca = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f67: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.56a: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.240: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.88a: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.345: %OrderedWith.assoc_type.88a = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.a81: %OrderedWith.assoc_type.88a = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.be8: %OrderedWith.assoc_type.88a = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.35d: %OrderedWith.assoc_type.88a = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.de2: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.f5e: %OrderedWith.type.240 = facet_value %Ordered, (%custom_witness.de2) [concrete]
// CHECK:STDOUT:   %pattern_type.04e: type = pattern_type %OrderedWith.type.240 [concrete]
// CHECK:STDOUT:   %complete_type.3b2: <witness> = complete_type_witness %OrderedWith.type.240 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.90a: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.f5e) [concrete]
// CHECK:STDOUT:   %.71b: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.90a, %OrderedWith.facet.f5e [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.31f: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.f5e) [concrete]
// CHECK:STDOUT:   %.1f0: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.31f, %OrderedWith.facet.f5e [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.622: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.f5e) [concrete]
// CHECK:STDOUT:   %.34e: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.622, %OrderedWith.facet.f5e [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.2a6: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.f5e) [concrete]
// CHECK:STDOUT:   %.eb8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.2a6, %OrderedWith.facet.f5e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.eb4: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.3ff)]
// CHECK:STDOUT:   %Core.import_ref.84c: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.760)]
// CHECK:STDOUT:   %Core.import_ref.01d: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.897)]
// CHECK:STDOUT:   %Core.import_ref.569: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.8bc)]
// CHECK:STDOUT:   %Core.import_ref.2d8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1)]
// CHECK:STDOUT:   %Core.import_ref.32f: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9defd1.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.984598.1)]
// CHECK:STDOUT:   %Core.import_ref.fb6: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1)]
// CHECK:STDOUT:   %Core.import_ref.5cd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.687390.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.4a995a.1)]
// CHECK:STDOUT:   %Core.import_ref.c33 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.79a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ea2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b4f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_27.2: @OrderedWith.loc13.%OrderedWith.type.loc13_48.1 (%OrderedWith.type.73dab8.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_48.1 [symbolic = %require_complete.loc15 (constants.%require_complete.a45)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2fabd7.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.687390.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_27.1 [symbolic = %.loc15_5.2 (constants.%.cb4)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.cb4) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc16_5.2 (constants.%.ac4)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.ac4) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element2, imports.%Core.import_ref.32f [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9defd1.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_27.1 [symbolic = %.loc17_5.2 (constants.%.994)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.994) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc18_5.2 (constants.%.eca)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.eca) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.eb4, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.cb4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.84c, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.ac4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.01d, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.994) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.569, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.eca) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.73dab8.2
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.10d
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.05039a.2
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.f5e) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.240
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%OrderedWith.facet.f5e
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.04e
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.56a
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.56a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.3b2
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.88a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.345
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.90a
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.71b
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.de2
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a81
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.31f
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.1f0
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.be8
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.622
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.34e
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.35d
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.2a6
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.eb8
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.73dab8.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d014af.1: %OrderedWith.type.73dab8.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.984598.1: %OrderedWith.WithSelf.Greater.type.9defd1.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1: %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.d014af.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.4a995a.1: %OrderedWith.WithSelf.Less.type.687390.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.73dab8.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.10d: type = pattern_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.73dab8.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2fabd7.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.7dbd55.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic]
// CHECK:STDOUT:   %assoc1.54de85.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic]
// CHECK:STDOUT:   %assoc2.3caaa9.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element2, imports.%Core.import_ref.32f [symbolic]
// CHECK:STDOUT:   %assoc3.4eb879.2: %OrderedWith.assoc_type.2fabd7.2 = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.05039a.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.a45: <witness> = require_complete_type %OrderedWith.type.73dab8.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3ff: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element0, imports.%Core.import_ref.c33 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.687390.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9defd1.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.cb4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.687390.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.cb4 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c89: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.760: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element1, imports.%Core.import_ref.79a [symbolic]
// CHECK:STDOUT:   %.ac4: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.ac4 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65b: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.897: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element2, imports.%Core.import_ref.ea2 [symbolic]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9defd1.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.994 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6dc: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.8bc: %OrderedWith.assoc_type.2fabd7.1 = assoc_entity element3, imports.%Core.import_ref.b4f [symbolic]
// CHECK:STDOUT:   %.eca: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.eca = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f67: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.e62: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %pattern_type.1e3: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %pattern_type.fab: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %A3: type = class_type @A3 [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %A3 [concrete]
// CHECK:STDOUT:   %B3: type = class_type @B3 [concrete]
// CHECK:STDOUT:   %pattern_type.a20: type = pattern_type %B3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.843: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.d9f: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.f28: %OrderedWith.assoc_type.d9f = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.b17: %OrderedWith.assoc_type.d9f = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.9ca: %OrderedWith.assoc_type.d9f = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.ac2: %OrderedWith.assoc_type.d9f = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6de: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.bb0: %OrderedWith.type.843 = facet_value %A1, (%custom_witness.6de) [concrete]
// CHECK:STDOUT:   %pattern_type.52e: type = pattern_type %OrderedWith.type.843 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.35b: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.555: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.15b: %OrderedWith.assoc_type.555 = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.ce2: %OrderedWith.assoc_type.555 = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.092: %OrderedWith.assoc_type.555 = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.fc9: %OrderedWith.assoc_type.555 = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffd: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.15d: %OrderedWith.type.35b = facet_value %B1, (%custom_witness.ffd) [concrete]
// CHECK:STDOUT:   %pattern_type.604: type = pattern_type %OrderedWith.type.35b [concrete]
// CHECK:STDOUT:   %OrderedWith.type.58a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2cf: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.d7b: %OrderedWith.assoc_type.2cf = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.c2a: %OrderedWith.assoc_type.2cf = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.20c: %OrderedWith.assoc_type.2cf = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.c0a: %OrderedWith.assoc_type.2cf = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.248: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.dfe: %OrderedWith.type.58a = facet_value %B2, (%custom_witness.248) [concrete]
// CHECK:STDOUT:   %pattern_type.2c4: type = pattern_type %OrderedWith.type.58a [concrete]
// CHECK:STDOUT:   %OrderedWith.type.d37: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.3d5: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.7d9: %OrderedWith.assoc_type.3d5 = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.f64: %OrderedWith.assoc_type.3d5 = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.060: %OrderedWith.assoc_type.3d5 = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.3e3: %OrderedWith.assoc_type.3d5 = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb5: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.647: %OrderedWith.type.d37 = facet_value %A3, (%custom_witness.eb5) [concrete]
// CHECK:STDOUT:   %pattern_type.f72: type = pattern_type %OrderedWith.type.d37 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.895: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1bb: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.19a: %OrderedWith.assoc_type.1bb = assoc_entity element0, imports.%Core.import_ref.5cd [concrete]
// CHECK:STDOUT:   %assoc1.573: %OrderedWith.assoc_type.1bb = assoc_entity element1, imports.%Core.import_ref.fb6 [concrete]
// CHECK:STDOUT:   %assoc2.a29: %OrderedWith.assoc_type.1bb = assoc_entity element2, imports.%Core.import_ref.32f [concrete]
// CHECK:STDOUT:   %assoc3.c40: %OrderedWith.assoc_type.1bb = assoc_entity element3, imports.%Core.import_ref.2d8 [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.63b: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.79f: %OrderedWith.type.895 = facet_value %B3, (%custom_witness.63b) [concrete]
// CHECK:STDOUT:   %pattern_type.970: type = pattern_type %OrderedWith.type.895 [concrete]
// CHECK:STDOUT:   %complete_type.c30: <witness> = complete_type_witness %OrderedWith.type.843 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.066: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.bb0) [concrete]
// CHECK:STDOUT:   %.dba: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.066, %OrderedWith.facet.bb0 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.a39: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.bb0) [concrete]
// CHECK:STDOUT:   %.0ef: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.a39, %OrderedWith.facet.bb0 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d68: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.bb0) [concrete]
// CHECK:STDOUT:   %.063: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d68, %OrderedWith.facet.bb0 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.90d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.bb0) [concrete]
// CHECK:STDOUT:   %.3a8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.90d, %OrderedWith.facet.bb0 [concrete]
// CHECK:STDOUT:   %complete_type.833: <witness> = complete_type_witness %OrderedWith.type.35b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.bac: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.15d) [concrete]
// CHECK:STDOUT:   %.7b9: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.bac, %OrderedWith.facet.15d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.263: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.15d) [concrete]
// CHECK:STDOUT:   %.107: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.263, %OrderedWith.facet.15d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.341: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.15d) [concrete]
// CHECK:STDOUT:   %.a86: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.341, %OrderedWith.facet.15d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9dd: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.15d) [concrete]
// CHECK:STDOUT:   %.e0e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9dd, %OrderedWith.facet.15d [concrete]
// CHECK:STDOUT:   %complete_type.b32: <witness> = complete_type_witness %OrderedWith.type.58a [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.242: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.dfe) [concrete]
// CHECK:STDOUT:   %.819: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.242, %OrderedWith.facet.dfe [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.bb1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.dfe) [concrete]
// CHECK:STDOUT:   %.984: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.bb1, %OrderedWith.facet.dfe [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.051: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.dfe) [concrete]
// CHECK:STDOUT:   %.bc8: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.051, %OrderedWith.facet.dfe [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.232: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.dfe) [concrete]
// CHECK:STDOUT:   %.9b5: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.232, %OrderedWith.facet.dfe [concrete]
// CHECK:STDOUT:   %complete_type.b8b: <witness> = complete_type_witness %OrderedWith.type.d37 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ab2: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.647) [concrete]
// CHECK:STDOUT:   %.06f: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ab2, %OrderedWith.facet.647 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.750: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.647) [concrete]
// CHECK:STDOUT:   %.914: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.750, %OrderedWith.facet.647 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.111: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.647) [concrete]
// CHECK:STDOUT:   %.a18: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.111, %OrderedWith.facet.647 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.a5d: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.647) [concrete]
// CHECK:STDOUT:   %.92d: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.a5d, %OrderedWith.facet.647 [concrete]
// CHECK:STDOUT:   %complete_type.b69: <witness> = complete_type_witness %OrderedWith.type.895 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.e74: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.79f) [concrete]
// CHECK:STDOUT:   %.4be: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.e74, %OrderedWith.facet.79f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.2de: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.79f) [concrete]
// CHECK:STDOUT:   %.79c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.2de, %OrderedWith.facet.79f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6fd: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.79f) [concrete]
// CHECK:STDOUT:   %.d57: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.6fd, %OrderedWith.facet.79f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.88b: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.79f) [concrete]
// CHECK:STDOUT:   %.cd5: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.88b, %OrderedWith.facet.79f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.eb4: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.3ff)]
// CHECK:STDOUT:   %Core.import_ref.84c: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.760)]
// CHECK:STDOUT:   %Core.import_ref.01d: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.897)]
// CHECK:STDOUT:   %Core.import_ref.569: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.8bc)]
// CHECK:STDOUT:   %Core.import_ref.2d8: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.38d1a4.1)]
// CHECK:STDOUT:   %Core.import_ref.32f: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9defd1.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.984598.1)]
// CHECK:STDOUT:   %Core.import_ref.fb6: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.c2ac22.1)]
// CHECK:STDOUT:   %Core.import_ref.5cd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.687390.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.4a995a.1)]
// CHECK:STDOUT:   %Core.import_ref.c33 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.79a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ea2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b4f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_27.2: @OrderedWith.loc54.%OrderedWith.type.loc54_48.1 (%OrderedWith.type.73dab8.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_48.1 [symbolic = %require_complete.loc56 (constants.%require_complete.a45)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2fabd7.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element0, imports.%Core.import_ref.5cd [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.687390.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_27.1 [symbolic = %.loc56_5.2 (constants.%.cb4)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.cb4) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element1, imports.%Core.import_ref.fb6 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8e175f.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc57_5.2 (constants.%.ac4)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.ac4) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element2, imports.%Core.import_ref.32f [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9defd1.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_27.1 [symbolic = %.loc58_5.2 (constants.%.994)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.994) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = assoc_entity element3, imports.%Core.import_ref.2d8 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a03cd9.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc59_5.2 (constants.%.eca)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.eca) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.eb4, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.7dbd55.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.cb4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.c89)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.84c, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.54de85.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.ac4) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.65b)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.01d, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.3caaa9.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.994) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.6dc)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.73dab8.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = specific_constant imports.%Core.import_ref.569, @OrderedWith.WithSelf(constants.%U, constants.%Self.d014af.1) [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2fabd7.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.4eb879.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.eca) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.f67)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.73dab8.2
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.10d
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.05039a.2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.bb0) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.843
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.bb0
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.52e
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c8a
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.e62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.c30
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.d9f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f28
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.066
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.dba
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.6de
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b17
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.a39
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.0ef
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.9ca
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.d68
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.063
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.ac2
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.90d
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.3a8
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.15d) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.35b
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.15d
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.604
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.e62
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.c8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.833
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.555
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.15b
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.bac
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.7b9
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.ffd
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ce2
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.263
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.107
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.092
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.341
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.a86
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.fc9
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9dd
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.e0e
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.dfe) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.58a
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.dfe
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.2c4
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.fab
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.1e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.b32
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.2cf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d7b
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.242
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.819
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.248
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c2a
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.bb1
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.984
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.20c
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.051
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.bc8
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.c0a
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.232
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.9b5
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.647) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.d37
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.647
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.f72
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.ada
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.a20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.b8b
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.3d5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7d9
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.ab2
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.06f
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.eb5
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f64
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.750
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.914
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.060
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.111
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.a18
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.3e3
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.a5d
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.92d
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.79f) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.895
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.79f
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.970
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.a20
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.ada
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.b69
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.1bb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.19a
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.e74
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.4be
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.63b
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.573
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.2de
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.79c
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.a29
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.6fd
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.d57
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.c40
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.88b
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.cd5
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
