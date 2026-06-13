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
  // CHECK:STDERR:   fn Less(self, other: Other) -> bool;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn LessOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn GreaterOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn Less(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn LessOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn Greater(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR:   fn GreaterOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7df555.1: %OrderedWith.type.32ae87.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c0b4fa.1: %OrderedWith.WithSelf.Greater.type.9ce25c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.e31723.1: %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.9e1f56.1: %OrderedWith.WithSelf.Less.type.b87f4b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.ebc: type = pattern_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %T.patt.aaa: %pattern_type.ebc = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.32ae87.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.c0023b.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic]
// CHECK:STDOUT:   %assoc1.3efd77.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic]
// CHECK:STDOUT:   %assoc2.48eff0.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element2, imports.%Core.import_ref.385 [symbolic]
// CHECK:STDOUT:   %assoc3.5d53f7.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element3, imports.%Core.import_ref.39b [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f6ec4f.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.a58: %pattern_type.f6ec4f.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.140: %pattern_type.f6ec4f.2 = at_binding_pattern x, %x.param_patt.a58 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.b85: %pattern_type.51d1c4.2 = at_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %require_complete.3c0: <witness> = require_complete_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %assoc0.13c: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element0, imports.%Core.import_ref.447 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.6ec: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b87f4b.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6ec = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.527: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.ee7: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element1, imports.%Core.import_ref.efc [symbolic]
// CHECK:STDOUT:   %.02d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.02d = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.001: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.131: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element2, imports.%Core.import_ref.e60 [symbolic]
// CHECK:STDOUT:   %.666: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9ce25c.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.666 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.735: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.18a: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element3, imports.%Core.import_ref.6c6 [symbolic]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.73b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.418: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.39d: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e83: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.621: %OrderedWith.assoc_type.e83 = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.a28: %OrderedWith.assoc_type.e83 = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.0b5: %OrderedWith.assoc_type.e83 = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.5e2: %OrderedWith.assoc_type.e83 = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.593: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.517: %OrderedWith.type.39d = facet_value %Ordered, (%custom_witness.593) [concrete]
// CHECK:STDOUT:   %pattern_type.d57: type = pattern_type %OrderedWith.type.39d [concrete]
// CHECK:STDOUT:   %T.patt.67e: %pattern_type.d57 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e3d: %pattern_type.e66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.51a: %pattern_type.e66 = at_binding_pattern x, %x.param_patt.e3d [concrete]
// CHECK:STDOUT:   %y.param_patt.4a3: %pattern_type.e66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.397: %pattern_type.e66 = at_binding_pattern y, %y.param_patt.4a3 [concrete]
// CHECK:STDOUT:   %complete_type.e48: <witness> = complete_type_witness %OrderedWith.type.39d [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.652: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.517) [concrete]
// CHECK:STDOUT:   %.92e: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.652, %OrderedWith.facet.517 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.b03: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.517) [concrete]
// CHECK:STDOUT:   %.765: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.b03, %OrderedWith.facet.517 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.2bc: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.517) [concrete]
// CHECK:STDOUT:   %.5ad: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.2bc, %OrderedWith.facet.517 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7fe: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.517) [concrete]
// CHECK:STDOUT:   %.6c0: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7fe, %OrderedWith.facet.517 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ef7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.13c)]
// CHECK:STDOUT:   %Core.import_ref.878: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.ee7)]
// CHECK:STDOUT:   %Core.import_ref.d4f: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.131)]
// CHECK:STDOUT:   %Core.import_ref.b87: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.18a)]
// CHECK:STDOUT:   %Core.import_ref.39b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1)]
// CHECK:STDOUT:   %Core.import_ref.385: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9ce25c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c0b4fa.1)]
// CHECK:STDOUT:   %Core.import_ref.3f7: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.e31723.1)]
// CHECK:STDOUT:   %Core.import_ref.e7a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.b87f4b.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.9e1f56.1)]
// CHECK:STDOUT:   %Core.import_ref.447 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.efc = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e60 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6c6 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_27.2: @OrderedWith.loc13.%OrderedWith.type.loc13_48.1 (%OrderedWith.type.32ae87.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_48.1 [symbolic = %require_complete.loc15 (constants.%require_complete.3c0)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.1d2e6f.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.b87f4b.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_27.1 [symbolic = %.loc15_5.2 (constants.%.6ec)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.6ec) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc16_5.2 (constants.%.02d)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.02d) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element2, imports.%Core.import_ref.385 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9ce25c.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_27.1 [symbolic = %.loc17_5.2 (constants.%.666)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.666) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element3, imports.%Core.import_ref.39b [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_27.1 [symbolic = %.loc18_5.2 (constants.%.73b)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.73b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_27.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.ef7, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.6ec) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.878, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.02d) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.d4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.666) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.b87, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.73b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc13_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.32ae87.2
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.ebc
// CHECK:STDOUT:   %T.patt.loc13_27.2 => constants.%T.patt.aaa
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.f6ec4f.2
// CHECK:STDOUT:   %x.param_patt.loc13_52.2 => constants.%x.param_patt.a58
// CHECK:STDOUT:   %x.patt.loc13_52.2 => constants.%x.patt.140
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc13_58.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc13_58.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.517) {
// CHECK:STDOUT:   %U.patt.loc13_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_48.1 => constants.%OrderedWith.type.39d
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.d57
// CHECK:STDOUT:   %T.patt.loc13_27.2 => constants.%T.patt.67e
// CHECK:STDOUT:   %T.loc13_27.1 => constants.%OrderedWith.facet.517
// CHECK:STDOUT:   %T.as_type.loc13_54.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_52 => constants.%pattern_type.e66
// CHECK:STDOUT:   %x.param_patt.loc13_52.2 => constants.%x.param_patt.e3d
// CHECK:STDOUT:   %x.patt.loc13_52.2 => constants.%x.patt.51a
// CHECK:STDOUT:   %pattern_type.loc13_58 => constants.%pattern_type.e66
// CHECK:STDOUT:   %y.param_patt.loc13_58.2 => constants.%y.param_patt.4a3
// CHECK:STDOUT:   %y.patt.loc13_58.2 => constants.%y.patt.397
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.e48
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.e83
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.621
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.652
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.92e
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.593
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a28
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.b03
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.765
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.0b5
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.2bc
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.5ad
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.5e2
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.7fe
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.6c0
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7df555.1: %OrderedWith.type.32ae87.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.c0b4fa.1: %OrderedWith.WithSelf.Greater.type.9ce25c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.e31723.1: %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.7df555.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.9e1f56.1: %OrderedWith.WithSelf.Less.type.b87f4b.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.32ae87.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.ebc: type = pattern_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %T.patt.aaa: %pattern_type.ebc = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.32ae87.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.1d2e6f.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.c0023b.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic]
// CHECK:STDOUT:   %assoc1.3efd77.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic]
// CHECK:STDOUT:   %assoc2.48eff0.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element2, imports.%Core.import_ref.385 [symbolic]
// CHECK:STDOUT:   %assoc3.5d53f7.2: %OrderedWith.assoc_type.1d2e6f.2 = assoc_entity element3, imports.%Core.import_ref.39b [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f6ec4f.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.a58: %pattern_type.f6ec4f.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.140: %pattern_type.f6ec4f.2 = at_binding_pattern x, %x.param_patt.a58 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.b85: %pattern_type.51d1c4.2 = at_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %require_complete.3c0: <witness> = require_complete_type %OrderedWith.type.32ae87.2 [symbolic]
// CHECK:STDOUT:   %assoc0.13c: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element0, imports.%Core.import_ref.447 [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b87f4b.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.9ce25c.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.6ec: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b87f4b.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.6ec = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.527: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.ee7: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element1, imports.%Core.import_ref.efc [symbolic]
// CHECK:STDOUT:   %.02db: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.02db = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.001: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.131: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element2, imports.%Core.import_ref.e60 [symbolic]
// CHECK:STDOUT:   %.666: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.9ce25c.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.666 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.735: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.18a: %OrderedWith.assoc_type.1d2e6f.1 = assoc_entity element3, imports.%Core.import_ref.6c6 [symbolic]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.73b = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.418: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.3b4: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.5b2: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %pattern_type.9f0: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %pattern_type.d1e: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %A3: type = class_type @A3 [concrete]
// CHECK:STDOUT:   %pattern_type.7fe: type = pattern_type %A3 [concrete]
// CHECK:STDOUT:   %B3: type = class_type @B3 [concrete]
// CHECK:STDOUT:   %pattern_type.2dd: type = pattern_type %B3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.4d9: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.993: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.9a4: %OrderedWith.assoc_type.993 = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.bc4: %OrderedWith.assoc_type.993 = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.da5: %OrderedWith.assoc_type.993 = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.17b: %OrderedWith.assoc_type.993 = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1d9: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.370: %OrderedWith.type.4d9 = facet_value %A1, (%custom_witness.1d9) [concrete]
// CHECK:STDOUT:   %pattern_type.53b: type = pattern_type %OrderedWith.type.4d9 [concrete]
// CHECK:STDOUT:   %T.patt.8c1: %pattern_type.53b = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.c7f: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.4ff: %pattern_type.3b4 = at_binding_pattern x, %x.param_patt.c7f [concrete]
// CHECK:STDOUT:   %y.param_patt.9d7: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.979: %pattern_type.5b2 = at_binding_pattern y, %y.param_patt.9d7 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.fe8: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.31d: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.a34: %OrderedWith.assoc_type.31d = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.27b: %OrderedWith.assoc_type.31d = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.b68: %OrderedWith.assoc_type.31d = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.833: %OrderedWith.assoc_type.31d = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d3a: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.288: %OrderedWith.type.fe8 = facet_value %B1, (%custom_witness.d3a) [concrete]
// CHECK:STDOUT:   %pattern_type.280: type = pattern_type %OrderedWith.type.fe8 [concrete]
// CHECK:STDOUT:   %T.patt.5de: %pattern_type.280 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.bff: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.f3d: %pattern_type.5b2 = at_binding_pattern x, %x.param_patt.bff [concrete]
// CHECK:STDOUT:   %y.param_patt.257: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.774: %pattern_type.3b4 = at_binding_pattern y, %y.param_patt.257 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.9ee: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.941: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.73a: %OrderedWith.assoc_type.941 = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.f9c: %OrderedWith.assoc_type.941 = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.f6a: %OrderedWith.assoc_type.941 = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.210: %OrderedWith.assoc_type.941 = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bbd: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.2ea: %OrderedWith.type.9ee = facet_value %B2, (%custom_witness.bbd) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %OrderedWith.type.9ee [concrete]
// CHECK:STDOUT:   %T.patt.e0a: %pattern_type.a96 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.0e6: %pattern_type.d1e = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.94e: %pattern_type.d1e = at_binding_pattern x, %x.param_patt.0e6 [concrete]
// CHECK:STDOUT:   %y.param_patt.a4f: %pattern_type.9f0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.912: %pattern_type.9f0 = at_binding_pattern y, %y.param_patt.a4f [concrete]
// CHECK:STDOUT:   %OrderedWith.type.cc8: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.45f: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.2c4: %OrderedWith.assoc_type.45f = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.219: %OrderedWith.assoc_type.45f = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.731: %OrderedWith.assoc_type.45f = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.6c4: %OrderedWith.assoc_type.45f = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.773: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.d6b: %OrderedWith.type.cc8 = facet_value %A3, (%custom_witness.773) [concrete]
// CHECK:STDOUT:   %pattern_type.edf: type = pattern_type %OrderedWith.type.cc8 [concrete]
// CHECK:STDOUT:   %T.patt.8a7: %pattern_type.edf = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.429: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.6e2: %pattern_type.7fe = at_binding_pattern x, %x.param_patt.429 [concrete]
// CHECK:STDOUT:   %y.param_patt.b51: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.7b8: %pattern_type.2dd = at_binding_pattern y, %y.param_patt.b51 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.30a: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.18e: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.d97: %OrderedWith.assoc_type.18e = assoc_entity element0, imports.%Core.import_ref.e7a [concrete]
// CHECK:STDOUT:   %assoc1.835: %OrderedWith.assoc_type.18e = assoc_entity element1, imports.%Core.import_ref.3f7 [concrete]
// CHECK:STDOUT:   %assoc2.27d: %OrderedWith.assoc_type.18e = assoc_entity element2, imports.%Core.import_ref.385 [concrete]
// CHECK:STDOUT:   %assoc3.6a9: %OrderedWith.assoc_type.18e = assoc_entity element3, imports.%Core.import_ref.39b [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.870: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.52f: %OrderedWith.type.30a = facet_value %B3, (%custom_witness.870) [concrete]
// CHECK:STDOUT:   %pattern_type.5db: type = pattern_type %OrderedWith.type.30a [concrete]
// CHECK:STDOUT:   %T.patt.36b: %pattern_type.5db = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.074: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.234: %pattern_type.2dd = at_binding_pattern x, %x.param_patt.074 [concrete]
// CHECK:STDOUT:   %y.param_patt.d86: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.fd9: %pattern_type.7fe = at_binding_pattern y, %y.param_patt.d86 [concrete]
// CHECK:STDOUT:   %complete_type.f5b: <witness> = complete_type_witness %OrderedWith.type.4d9 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.bf4: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.370) [concrete]
// CHECK:STDOUT:   %.e3b: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.bf4, %OrderedWith.facet.370 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.8d0: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.370) [concrete]
// CHECK:STDOUT:   %.736: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.8d0, %OrderedWith.facet.370 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.130: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.370) [concrete]
// CHECK:STDOUT:   %.b93: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.130, %OrderedWith.facet.370 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.1e3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.370) [concrete]
// CHECK:STDOUT:   %.ccd: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.1e3, %OrderedWith.facet.370 [concrete]
// CHECK:STDOUT:   %complete_type.9ae: <witness> = complete_type_witness %OrderedWith.type.fe8 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.d52: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.288) [concrete]
// CHECK:STDOUT:   %.367: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.d52, %OrderedWith.facet.288 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.f69: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.288) [concrete]
// CHECK:STDOUT:   %.0f8: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.f69, %OrderedWith.facet.288 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.f1e: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.288) [concrete]
// CHECK:STDOUT:   %.2f6: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.f1e, %OrderedWith.facet.288 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.f75: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.288) [concrete]
// CHECK:STDOUT:   %.3fa: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.f75, %OrderedWith.facet.288 [concrete]
// CHECK:STDOUT:   %complete_type.674: <witness> = complete_type_witness %OrderedWith.type.9ee [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.b7e: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.02d8: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.b7e, %OrderedWith.facet.2ea [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.1dc: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.fd7: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.1dc, %OrderedWith.facet.2ea [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.693: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.380: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.693, %OrderedWith.facet.2ea [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.906: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.2ea) [concrete]
// CHECK:STDOUT:   %.b8f: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.906, %OrderedWith.facet.2ea [concrete]
// CHECK:STDOUT:   %complete_type.672: <witness> = complete_type_witness %OrderedWith.type.cc8 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.c08: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.a95: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.c08, %OrderedWith.facet.d6b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.f02: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.fda: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.f02, %OrderedWith.facet.d6b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d28: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.24d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d28, %OrderedWith.facet.d6b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.5ab: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.898: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.5ab, %OrderedWith.facet.d6b [concrete]
// CHECK:STDOUT:   %complete_type.04a: <witness> = complete_type_witness %OrderedWith.type.30a [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.9d3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.52f) [concrete]
// CHECK:STDOUT:   %.fb6: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.9d3, %OrderedWith.facet.52f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.ff1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.52f) [concrete]
// CHECK:STDOUT:   %.bb5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.ff1, %OrderedWith.facet.52f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.d71: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.52f) [concrete]
// CHECK:STDOUT:   %.51d: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.d71, %OrderedWith.facet.52f [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.ec1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.52f) [concrete]
// CHECK:STDOUT:   %.4c7: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.ec1, %OrderedWith.facet.52f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ef7: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.13c)]
// CHECK:STDOUT:   %Core.import_ref.878: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.ee7)]
// CHECK:STDOUT:   %Core.import_ref.d4f: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.131)]
// CHECK:STDOUT:   %Core.import_ref.b87: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.18a)]
// CHECK:STDOUT:   %Core.import_ref.39b: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.937fbb.1)]
// CHECK:STDOUT:   %Core.import_ref.385: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.9ce25c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.c0b4fa.1)]
// CHECK:STDOUT:   %Core.import_ref.3f7: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.e31723.1)]
// CHECK:STDOUT:   %Core.import_ref.e7a: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.b87f4b.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.9e1f56.1)]
// CHECK:STDOUT:   %Core.import_ref.447 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.efc = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e60 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6c6 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_27.2: @OrderedWith.loc54.%OrderedWith.type.loc54_48.1 (%OrderedWith.type.32ae87.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_48.1 [symbolic = %require_complete.loc56 (constants.%require_complete.3c0)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.1d2e6f.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element0, imports.%Core.import_ref.e7a [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.b87f4b.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_27.1 [symbolic = %.loc56_5.2 (constants.%.6ec)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_27.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.6ec) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element1, imports.%Core.import_ref.3f7 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.13d375.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc57_5.2 (constants.%.02db)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.02db) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element2, imports.%Core.import_ref.385 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.9ce25c.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_27.1 [symbolic = %.loc58_5.2 (constants.%.666)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.666) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = assoc_entity element3, imports.%Core.import_ref.39b [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.7dedb2.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_27.1 [symbolic = %.loc59_5.2 (constants.%.73b)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.73b) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_27.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.ef7, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.c0023b.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.6ec) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.527)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.878, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.3efd77.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.02db) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.001)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.d4f, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.48eff0.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.666) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.735)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_54.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_48.1 (constants.%OrderedWith.type.32ae87.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = specific_constant imports.%Core.import_ref.b87, @OrderedWith.WithSelf(constants.%U, constants.%Self.7df555.1) [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.1d2e6f.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.5d53f7.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.73b) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.32ae87.2
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.ebc
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.aaa
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.f6ec4f.2
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.a58
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.140
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.370) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.4d9
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.53b
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.8c1
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.370
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.c7f
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.4ff
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.9d7
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.979
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.f5b
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.993
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9a4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.bf4
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.e3b
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.1d9
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.bc4
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.8d0
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.736
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.da5
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.130
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.b93
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.17b
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.1e3
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.ccd
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.288) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.fe8
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.280
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.5de
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.288
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.bff
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.f3d
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.257
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.774
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.9ae
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.31d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a34
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.d52
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.367
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.d3a
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.27b
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.f69
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.0f8
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.b68
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.f1e
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.2f6
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.833
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.f75
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.3fa
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.2ea) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.9ee
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.a96
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.e0a
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.2ea
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.d1e
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.0e6
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.94e
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.9f0
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.a4f
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.912
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.674
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.941
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.73a
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.b7e
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.02d8
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.bbd
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f9c
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.1dc
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.fd7
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.f6a
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.693
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.380
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.210
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.906
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.b8f
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.d6b) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.cc8
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.edf
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.8a7
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.d6b
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.429
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.6e2
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.b51
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.7b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.672
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.45f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2c4
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.c08
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.a95
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.773
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.219
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.f02
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.fda
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.731
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.d28
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.24d
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.6c4
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.5ab
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.898
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.52f) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_48.1 => constants.%OrderedWith.type.30a
// CHECK:STDOUT:   %pattern_type.loc54_27 => constants.%pattern_type.5db
// CHECK:STDOUT:   %T.patt.loc54_27.2 => constants.%T.patt.36b
// CHECK:STDOUT:   %T.loc54_27.1 => constants.%OrderedWith.facet.52f
// CHECK:STDOUT:   %T.as_type.loc54_54.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_52 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %x.param_patt.loc54_52.2 => constants.%x.param_patt.074
// CHECK:STDOUT:   %x.patt.loc54_52.2 => constants.%x.patt.234
// CHECK:STDOUT:   %pattern_type.loc54_58 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %y.param_patt.loc54_58.2 => constants.%y.param_patt.d86
// CHECK:STDOUT:   %y.patt.loc54_58.2 => constants.%y.patt.fd9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_52 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_58 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.04a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.18e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d97
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.9d3
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.fb6
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.870
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.835
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.ff1
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.bb5
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.27d
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.d71
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.51d
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.6a9
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.ec1
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.4c7
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
