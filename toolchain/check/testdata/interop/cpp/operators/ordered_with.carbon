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
// CHECK:STDOUT:   %OrderedWith.type.1ce6d7.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5899f7.1: %OrderedWith.type.1ce6d7.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72f2ab.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.433a82.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.8e8d9f.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.4b40f6.1: %OrderedWith.WithSelf.Greater.type.8e8d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.cf3f1c.1: %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0103c6.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.828cee.1: %OrderedWith.WithSelf.Less.type.0103c6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.1ce6d7.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9c4: type = pattern_type %OrderedWith.type.1ce6d7.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.1ce6d7.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72f2ab.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.76ced5.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element0, imports.%Core.import_ref.250 [symbolic]
// CHECK:STDOUT:   %assoc1.248efd.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element1, imports.%Core.import_ref.af2 [symbolic]
// CHECK:STDOUT:   %assoc2.cac9ed.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element2, imports.%Core.import_ref.ece [symbolic]
// CHECK:STDOUT:   %assoc3.9908c2.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element3, imports.%Core.import_ref.bcf [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.2e7263.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.028: <witness> = require_complete_type %OrderedWith.type.1ce6d7.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d47: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element0, imports.%Core.import_ref.516 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0103c6.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.8e8d9f.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.e72: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.0103c6.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e72 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fd5: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.d74: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element1, imports.%Core.import_ref.785 [symbolic]
// CHECK:STDOUT:   %.951: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.951 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.315: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.4ac: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element2, imports.%Core.import_ref.3b8 [symbolic]
// CHECK:STDOUT:   %.813: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.8e8d9f.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.813 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9d6: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.89a: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element3, imports.%Core.import_ref.29c [symbolic]
// CHECK:STDOUT:   %.137: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.137 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.261: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.56a: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.ef0: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.298: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.0d2: %OrderedWith.assoc_type.298 = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.d8d: %OrderedWith.assoc_type.298 = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.1b7: %OrderedWith.assoc_type.298 = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.1e4: %OrderedWith.assoc_type.298 = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.99d: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.0d3: %OrderedWith.type.ef0 = facet_value %Ordered, (%custom_witness.99d) [concrete]
// CHECK:STDOUT:   %pattern_type.432: type = pattern_type %OrderedWith.type.ef0 [concrete]
// CHECK:STDOUT:   %complete_type.831: <witness> = complete_type_witness %OrderedWith.type.ef0 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.d4c: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.0d3) [concrete]
// CHECK:STDOUT:   %.2f2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.d4c, %OrderedWith.facet.0d3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.e37: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.0d3) [concrete]
// CHECK:STDOUT:   %.bcc: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.e37, %OrderedWith.facet.0d3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.2d4: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.0d3) [concrete]
// CHECK:STDOUT:   %.287: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.2d4, %OrderedWith.facet.0d3 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.589: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.0d3) [concrete]
// CHECK:STDOUT:   %.e4e: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.589, %OrderedWith.facet.0d3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.08c: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d47)]
// CHECK:STDOUT:   %Core.import_ref.6b7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.d74)]
// CHECK:STDOUT:   %Core.import_ref.bc5: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.4ac)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.89a)]
// CHECK:STDOUT:   %Core.import_ref.bcf: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.433a82.1)]
// CHECK:STDOUT:   %Core.import_ref.ece: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.8e8d9f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.4b40f6.1)]
// CHECK:STDOUT:   %Core.import_ref.af2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.cf3f1c.1)]
// CHECK:STDOUT:   %Core.import_ref.250: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.0103c6.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.828cee.1)]
// CHECK:STDOUT:   %Core.import_ref.516 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.785 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3b8 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.29c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_27.2: @OrderedWith.loc13.%OrderedWith.type.loc13_48.1 (%OrderedWith.type.1ce6d7.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_48.1 [symbolic = %require_complete.loc15 (constants.%require_complete.028)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.72f2ab.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element0, imports.%Core.import_ref.250 [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.0103c6.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_27.1 [symbolic = %.loc15_5.2 (constants.%.e72)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.e72) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.fd5)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element1, imports.%Core.import_ref.af2 [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc16_5.2 (constants.%.951)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.951) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.315)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element2, imports.%Core.import_ref.ece [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.8e8d9f.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_27.1 [symbolic = %.loc17_5.2 (constants.%.813)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.813) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.9d6)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element3, imports.%Core.import_ref.bcf [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc18_5.2 (constants.%.137)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.137) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.08c, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.e72) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.fd5)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.6b7, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.951) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.315)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.bc5, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.813) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.9d6)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.cfb, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.137) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.1ce6d7.2
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.9c4
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.2e7263.2
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.0d3) {
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.ef0
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%OrderedWith.facet.0d3
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.432
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.56a
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.56a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.831
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.298
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0d2
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.d4c
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.2f2
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.99d
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d8d
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.e37
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.bcc
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.1b7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.2d4
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.287
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.1e4
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.589
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.e4e
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.1ce6d7.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5899f7.1: %OrderedWith.type.1ce6d7.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72f2ab.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.433a82.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.8e8d9f.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.4b40f6.1: %OrderedWith.WithSelf.Greater.type.8e8d9f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.cf3f1c.1: %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0103c6.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.5899f7.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.828cee.1: %OrderedWith.WithSelf.Less.type.0103c6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.1ce6d7.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9c4: type = pattern_type %OrderedWith.type.1ce6d7.2 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.1ce6d7.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.72f2ab.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.76ced5.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element0, imports.%Core.import_ref.250 [symbolic]
// CHECK:STDOUT:   %assoc1.248efd.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element1, imports.%Core.import_ref.af2 [symbolic]
// CHECK:STDOUT:   %assoc2.cac9ed.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element2, imports.%Core.import_ref.ece [symbolic]
// CHECK:STDOUT:   %assoc3.9908c2.2: %OrderedWith.assoc_type.72f2ab.2 = assoc_entity element3, imports.%Core.import_ref.bcf [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.2e7263.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.028: <witness> = require_complete_type %OrderedWith.type.1ce6d7.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d47: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element0, imports.%Core.import_ref.516 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.0103c6.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.8e8d9f.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.e72: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.0103c6.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e72 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fd5: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.d74: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element1, imports.%Core.import_ref.785 [symbolic]
// CHECK:STDOUT:   %.951: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.951 = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.315: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.4ac: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element2, imports.%Core.import_ref.3b8 [symbolic]
// CHECK:STDOUT:   %.813: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.8e8d9f.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.813 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9d6: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.89a: %OrderedWith.assoc_type.72f2ab.1 = assoc_entity element3, imports.%Core.import_ref.29c [symbolic]
// CHECK:STDOUT:   %.137: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.137 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.261: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
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
// CHECK:STDOUT:   %OrderedWith.type.e85: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.46d: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.074: %OrderedWith.assoc_type.46d = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.b94: %OrderedWith.assoc_type.46d = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.085: %OrderedWith.assoc_type.46d = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.426: %OrderedWith.assoc_type.46d = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ca6: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.97c: %OrderedWith.type.e85 = facet_value %A1, (%custom_witness.ca6) [concrete]
// CHECK:STDOUT:   %pattern_type.187: type = pattern_type %OrderedWith.type.e85 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.de5: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.6f1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.5e4: %OrderedWith.assoc_type.6f1 = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.547: %OrderedWith.assoc_type.6f1 = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.e9f: %OrderedWith.assoc_type.6f1 = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.c50: %OrderedWith.assoc_type.6f1 = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.998: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.53d: %OrderedWith.type.de5 = facet_value %B1, (%custom_witness.998) [concrete]
// CHECK:STDOUT:   %pattern_type.18c: type = pattern_type %OrderedWith.type.de5 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b2b: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.ede: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.ae2: %OrderedWith.assoc_type.ede = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.a64: %OrderedWith.assoc_type.ede = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.24c: %OrderedWith.assoc_type.ede = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.f57: %OrderedWith.assoc_type.ede = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a76: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.f05: %OrderedWith.type.b2b = facet_value %B2, (%custom_witness.a76) [concrete]
// CHECK:STDOUT:   %pattern_type.c18: type = pattern_type %OrderedWith.type.b2b [concrete]
// CHECK:STDOUT:   %OrderedWith.type.3b2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.b1f: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.635: %OrderedWith.assoc_type.b1f = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.e4a: %OrderedWith.assoc_type.b1f = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.21a: %OrderedWith.assoc_type.b1f = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.1bb: %OrderedWith.assoc_type.b1f = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.223: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.9e9: %OrderedWith.type.3b2 = facet_value %A3, (%custom_witness.223) [concrete]
// CHECK:STDOUT:   %pattern_type.448: type = pattern_type %OrderedWith.type.3b2 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.708: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.2d1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.cb0: %OrderedWith.assoc_type.2d1 = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %assoc1.aa4: %OrderedWith.assoc_type.2d1 = assoc_entity element1, imports.%Core.import_ref.af2 [concrete]
// CHECK:STDOUT:   %assoc2.ec4: %OrderedWith.assoc_type.2d1 = assoc_entity element2, imports.%Core.import_ref.ece [concrete]
// CHECK:STDOUT:   %assoc3.781: %OrderedWith.assoc_type.2d1 = assoc_entity element3, imports.%Core.import_ref.bcf [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c0: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.725: %OrderedWith.type.708 = facet_value %B3, (%custom_witness.0c0) [concrete]
// CHECK:STDOUT:   %pattern_type.2c5: type = pattern_type %OrderedWith.type.708 [concrete]
// CHECK:STDOUT:   %complete_type.979: <witness> = complete_type_witness %OrderedWith.type.e85 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.bd5: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.97c) [concrete]
// CHECK:STDOUT:   %.2ff: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.bd5, %OrderedWith.facet.97c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.651: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.97c) [concrete]
// CHECK:STDOUT:   %.b55: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.651, %OrderedWith.facet.97c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d8f: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.97c) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d8f, %OrderedWith.facet.97c [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.d67: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.97c) [concrete]
// CHECK:STDOUT:   %.a97: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.d67, %OrderedWith.facet.97c [concrete]
// CHECK:STDOUT:   %complete_type.409: <witness> = complete_type_witness %OrderedWith.type.de5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.6e7: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.53d) [concrete]
// CHECK:STDOUT:   %.866: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.6e7, %OrderedWith.facet.53d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.03c: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.53d) [concrete]
// CHECK:STDOUT:   %.87e: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.03c, %OrderedWith.facet.53d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.829: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.53d) [concrete]
// CHECK:STDOUT:   %.bb6: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.829, %OrderedWith.facet.53d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.328: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.53d) [concrete]
// CHECK:STDOUT:   %.d08: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.328, %OrderedWith.facet.53d [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %OrderedWith.type.b2b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c19: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.f05) [concrete]
// CHECK:STDOUT:   %.4b8: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c19, %OrderedWith.facet.f05 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3f7: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.f05) [concrete]
// CHECK:STDOUT:   %.12b: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3f7, %OrderedWith.facet.f05 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.0a3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.f05) [concrete]
// CHECK:STDOUT:   %.3c2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.0a3, %OrderedWith.facet.f05 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.04f: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.f05) [concrete]
// CHECK:STDOUT:   %.ec8: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.04f, %OrderedWith.facet.f05 [concrete]
// CHECK:STDOUT:   %complete_type.0be: <witness> = complete_type_witness %OrderedWith.type.3b2 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.cb6: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.9e9) [concrete]
// CHECK:STDOUT:   %.74c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.cb6, %OrderedWith.facet.9e9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.96c: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.9e9) [concrete]
// CHECK:STDOUT:   %.4c8: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.96c, %OrderedWith.facet.9e9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.6e1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.9e9) [concrete]
// CHECK:STDOUT:   %.685: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.6e1, %OrderedWith.facet.9e9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.886: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.9e9) [concrete]
// CHECK:STDOUT:   %.d61: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.886, %OrderedWith.facet.9e9 [concrete]
// CHECK:STDOUT:   %complete_type.9fb: <witness> = complete_type_witness %OrderedWith.type.708 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c16: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.725) [concrete]
// CHECK:STDOUT:   %.4d3: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c16, %OrderedWith.facet.725 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.c04: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.725) [concrete]
// CHECK:STDOUT:   %.019: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.c04, %OrderedWith.facet.725 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.112: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.725) [concrete]
// CHECK:STDOUT:   %.d65: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.112, %OrderedWith.facet.725 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.566: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.725) [concrete]
// CHECK:STDOUT:   %.9d0: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.566, %OrderedWith.facet.725 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.08c: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d47)]
// CHECK:STDOUT:   %Core.import_ref.6b7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.d74)]
// CHECK:STDOUT:   %Core.import_ref.bc5: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.4ac)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.89a)]
// CHECK:STDOUT:   %Core.import_ref.bcf: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.433a82.1)]
// CHECK:STDOUT:   %Core.import_ref.ece: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.8e8d9f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.4b40f6.1)]
// CHECK:STDOUT:   %Core.import_ref.af2: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.cf3f1c.1)]
// CHECK:STDOUT:   %Core.import_ref.250: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.0103c6.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.828cee.1)]
// CHECK:STDOUT:   %Core.import_ref.516 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.785 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3b8 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.29c = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_27.2: @OrderedWith.loc54.%OrderedWith.type.loc54_48.1 (%OrderedWith.type.1ce6d7.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_48.1 [symbolic = %require_complete.loc56 (constants.%require_complete.028)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.72f2ab.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element0, imports.%Core.import_ref.250 [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.0103c6.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_27.1 [symbolic = %.loc56_5.2 (constants.%.e72)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.e72) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.fd5)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element1, imports.%Core.import_ref.af2 [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8f86f8.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc57_5.2 (constants.%.951)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.951) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.315)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element2, imports.%Core.import_ref.ece [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.8e8d9f.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_27.1 [symbolic = %.loc58_5.2 (constants.%.813)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.813) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.9d6)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = assoc_entity element3, imports.%Core.import_ref.bcf [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.cd7ce3.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc59_5.2 (constants.%.137)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.137) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.08c, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.76ced5.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.e72) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.fd5)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.6b7, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.248efd.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.951) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.315)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.bc5, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.cac9ed.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.813) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.9d6)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.1ce6d7.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = specific_constant imports.%Core.import_ref.cfb, @OrderedWith.WithSelf(constants.%U, constants.%Self.5899f7.1) [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.72f2ab.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.9908c2.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.137) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.261)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.1ce6d7.2
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.9c4
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.2e7263.2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.97c) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.e85
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.97c
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.187
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.c8a
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.e62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.979
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.46d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.074
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.bd5
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.2ff
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.ca6
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b94
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.651
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.b55
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.085
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.d8f
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.68a
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.426
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.d67
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.a97
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.53d) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.de5
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.53d
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.18c
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.e62
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.c8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.409
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.6f1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5e4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.6e7
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.866
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.998
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.547
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.03c
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.87e
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.e9f
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.829
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.bb6
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.c50
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.328
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.d08
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.f05) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.b2b
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.f05
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.c18
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.fab
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.1e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.65d
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.ede
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ae2
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.c19
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.4b8
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.a76
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a64
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.3f7
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.12b
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.24c
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.0a3
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.3c2
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.f57
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.04f
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.ec8
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.9e9) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.3b2
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.9e9
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.448
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.ada
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.a20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.0be
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.b1f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.635
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.cb6
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.74c
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.223
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e4a
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.96c
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.4c8
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.21a
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.6e1
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.685
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.1bb
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.886
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.d61
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.725) {
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.708
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.725
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.2c5
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.a20
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.ada
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.9fb
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.2d1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cb0
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.c16
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.4d3
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.0c0
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.aa4
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.c04
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.019
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.ec4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.112
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.d65
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.781
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.566
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.9d0
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
