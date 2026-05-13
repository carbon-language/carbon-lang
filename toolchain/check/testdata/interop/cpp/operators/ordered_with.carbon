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
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.356: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.03f: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.5a7: %OrderedWith.assoc_type.03f = assoc_entity element0, imports.%Core.import_ref.255 [concrete]
// CHECK:STDOUT:   %assoc1.dce: %OrderedWith.assoc_type.03f = assoc_entity element1, imports.%Core.import_ref.857 [concrete]
// CHECK:STDOUT:   %assoc2.36d: %OrderedWith.assoc_type.03f = assoc_entity element2, imports.%Core.import_ref.a0a [concrete]
// CHECK:STDOUT:   %assoc3.00d: %OrderedWith.assoc_type.03f = assoc_entity element3, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.756: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.251: %OrderedWith.type.356 = facet_value %Ordered, (%custom_witness.756) [concrete]
// CHECK:STDOUT:   %pattern_type.86e: type = pattern_type %OrderedWith.type.356 [concrete]
// CHECK:STDOUT:   %complete_type.18e: <witness> = complete_type_witness %OrderedWith.type.356 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ce4: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.251) [concrete]
// CHECK:STDOUT:   %.e21: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ce4, %OrderedWith.facet.251 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.73b: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.251) [concrete]
// CHECK:STDOUT:   %.4af: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.73b, %OrderedWith.facet.251 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.cb0: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.251) [concrete]
// CHECK:STDOUT:   %.6fb: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.cb0, %OrderedWith.facet.251 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.df2: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.251) [concrete]
// CHECK:STDOUT:   %.c15: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.df2, %OrderedWith.facet.251 [concrete]
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
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_27.2: @OrderedWith.loc13.%OrderedWith.type.loc13_48.1 (%OrderedWith.type.ca8786.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_48.1 [symbolic = %require_complete.loc15 (constants.%require_complete.3d2)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2b366e.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element0, imports.%Core.import_ref.255 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.df27e9.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_27.1 [symbolic = %.loc15_5.2 (constants.%.0af)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.0af) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element1, imports.%Core.import_ref.857 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.970504.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc16_5.2 (constants.%.81b)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.81b) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.6f28f1.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_27.1 [symbolic = %.loc17_5.2 (constants.%.f69)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.f69) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc18_5.2 (constants.%.206)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.206) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.7e9, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.0af) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.c86, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.81b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.f4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.f69) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.065, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.206) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.ca8786.2
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.d3f
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.c70b34.2
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.251) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.356
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%OrderedWith.facet.251
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.86e
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.88f
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.88f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.18e
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.03f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5a7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.ce4
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.e21
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.756
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.dce
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.73b
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.4af
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.36d
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.cb0
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.6fb
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.00d
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.df2
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.c15
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
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
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_27.2: @OrderedWith.loc54.%OrderedWith.type.loc54_48.1 (%OrderedWith.type.ca8786.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_48.1 [symbolic = %require_complete.loc56 (constants.%require_complete.3d2)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.2b366e.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element0, imports.%Core.import_ref.255 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.df27e9.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_27.1 [symbolic = %.loc56_5.2 (constants.%.0af)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.0af) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element1, imports.%Core.import_ref.857 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.970504.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc57_5.2 (constants.%.81b)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.81b) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element2, imports.%Core.import_ref.a0a [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.6f28f1.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_27.1 [symbolic = %.loc58_5.2 (constants.%.f69)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.f69) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = assoc_entity element3, imports.%Core.import_ref.bb2 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9696d2.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc59_5.2 (constants.%.206)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.206) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.7e9, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.f6e965.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.0af) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.072)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.c86, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.23402c.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.81b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.2f8)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.f4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.771acd.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.f69) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.1cd)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.ca8786.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = specific_constant imports.%Core.import_ref.065, @OrderedWith.WithSelf(constants.%U, constants.%Self.adc9f7.1) [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.2b366e.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.ba2aca.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.206) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.b2f)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.ca8786.2
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.d3f
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c70b34.2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.0fa) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.773
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.0fa
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.f4b
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.57d
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.c01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.68a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.5e8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d03
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.e0f
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.e26
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.e39
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f6b
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.15d
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.c48
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.911
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.3a1
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.dba
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.a71
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.ccd
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.64a
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.d5c) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.cf0
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.d5c
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.3f7
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c01
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.57d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.ee8
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.42c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4b4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.f2f
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.d99
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.b81
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e14
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.64e
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.d5b
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.733
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.4af
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.b8a
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.ffa
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.037
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.edc
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.edc) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.26a
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.edc
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.899
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.fe2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.6ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.518
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.688
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a70
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.c1b
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.ff0
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.176
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.54d
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.341
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.d0f
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.15d
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.bb0
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.e4f
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.68c
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.c69
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.fe6
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.ba3) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.232
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.ba3
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.f06
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.bb4
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.b85
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.7ba
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.02d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0d8
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.3f8
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.7f9
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.499
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f4d
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.a19
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.d93
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.ba7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.ae5
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.152
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.fa1
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.17e
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.4d1
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.b33) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.343
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.b33
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.e5d
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.b85
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.bb4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.af4
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.8f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4c0
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.444
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.5ac
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.679
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c91
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.e9d
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.158
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.7b7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.7e9
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.8ef
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.1c2
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.408
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.002
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
