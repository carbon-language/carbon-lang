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

fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
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

fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:28:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-36]]:8: error: cannot implicitly convert expression of type `Core.Optional(i32* as Core.OptionalStorage)` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<=(LessEqualReturnsBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-39]]:8: note: type `Core.Optional(i32* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<=(LessEqualReturnsBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:29:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn LessOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+69]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-39]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-42]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>=(GreaterEqualReturnsBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:31:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn GreaterOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+56]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-37]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-40]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:28:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Less(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-49]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator<=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-52]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator<=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:29:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn LessOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+30]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-61]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-64]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:30:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Greater(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-73]]:8: error: cannot implicitly convert expression of type `f64` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator>=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-76]]:8: note: type `f64` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator>=(AllReturnBoolish) const -> double;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:31:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn GreaterOrEquivalent(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
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

fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less, missing_less);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingLessEqual` into type implementing `Core.OrderedWith(Cpp.MissingLessEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_less_equal, missing_less_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-26]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_less_equal, missing_less_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreater` into type implementing `Core.OrderedWith(Cpp.MissingGreater)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater, missing_greater);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater, missing_greater);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.MissingGreaterEqual` into type implementing `Core.OrderedWith(Cpp.MissingGreaterEqual)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(missing_greater_equal, missing_greater_equal);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-44]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(missing_greater_equal, missing_greater_equal);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoRelevantMembers` into type implementing `Core.OrderedWith(Cpp.NoRelevantMembers)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(no_relevant_members, no_relevant_members);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-53]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(no_relevant_members, no_relevant_members);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.LessReturnsVoid` into type implementing `Core.OrderedWith(Cpp.LessReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   OrderedWith(less_returns_void, less_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-62]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn OrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  OrderedWith(less_returns_void, less_returns_void);

  OrderedWith(less_non_const, less_non_const);
}

// CHECK:STDOUT: --- returns_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.635006.1: %OrderedWith.type.507fad.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.bb34ec.1: %OrderedWith.WithSelf.Greater.type.947541.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.ac2301.1: %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.51e833.1: %OrderedWith.WithSelf.Less.type.104f17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.88a: type = pattern_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %T.patt.b1d: %pattern_type.88a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.507fad.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f17d83.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1a7: %pattern_type.f17d83.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.708: %pattern_type.f17d83.2 = wrapper_binding_pattern x, %x.param_patt.1a7 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.2 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.10837a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic]
// CHECK:STDOUT:   %assoc1.4f4378.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic]
// CHECK:STDOUT:   %assoc2.60074a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic]
// CHECK:STDOUT:   %assoc3.4e66f6.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic]
// CHECK:STDOUT:   %require_complete.f6d: <witness> = require_complete_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d91: %OrderedWith.assoc_type.06a530.1 = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.236: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.104f17.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.236 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.15a: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.2e2: %OrderedWith.assoc_type.06a530.1 = assoc_entity element1, imports.%Core.import_ref.80a [symbolic]
// CHECK:STDOUT:   %.0de: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.0de = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1d5: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.935: %OrderedWith.assoc_type.06a530.1 = assoc_entity element2, imports.%Core.import_ref.fe3 [symbolic]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.947541.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.5e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.913: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.dfc: %OrderedWith.assoc_type.06a530.1 = assoc_entity element3, imports.%Core.import_ref.1f1 [symbolic]
// CHECK:STDOUT:   %.489: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.489 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e09: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %Ordered: type = class_type @Ordered [concrete]
// CHECK:STDOUT:   %pattern_type.e66: type = pattern_type %Ordered [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %OrderedWith.type.431: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Ordered)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e03: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %assoc0.c6f: %OrderedWith.assoc_type.e03 = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.965: %OrderedWith.assoc_type.e03 = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.b19: %OrderedWith.assoc_type.e03 = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.673: %OrderedWith.assoc_type.e03 = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %Ordered.Less.type: type = fn_type @Ordered.Less [concrete]
// CHECK:STDOUT:   %Ordered.Less: %Ordered.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent.type: type = fn_type @Ordered.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.LessOrEquivalent: %Ordered.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.Greater.type: type = fn_type @Ordered.Greater [concrete]
// CHECK:STDOUT:   %Ordered.Greater: %Ordered.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent.type: type = fn_type @Ordered.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %Ordered.GreaterOrEquivalent: %Ordered.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c96: <witness> = custom_witness (%Ordered.Less, %Ordered.LessOrEquivalent, %Ordered.Greater, %Ordered.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%Ordered) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.b02: %OrderedWith.type.431 = facet_value %Ordered, (%custom_witness.c96) [concrete]
// CHECK:STDOUT:   %pattern_type.c99: type = pattern_type %OrderedWith.type.431 [concrete]
// CHECK:STDOUT:   %T.patt.264: %pattern_type.c99 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e3d: %pattern_type.e66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.88c: %pattern_type.e66 = wrapper_binding_pattern x, %x.param_patt.e3d [concrete]
// CHECK:STDOUT:   %y.param_patt.4a3: %pattern_type.e66 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.68b: %pattern_type.e66 = wrapper_binding_pattern y, %y.param_patt.4a3 [concrete]
// CHECK:STDOUT:   %complete_type.23a: <witness> = complete_type_witness %OrderedWith.type.431 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.ae1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.b02) [concrete]
// CHECK:STDOUT:   %.05c: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.ae1, %OrderedWith.facet.b02 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.cbc: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.b02) [concrete]
// CHECK:STDOUT:   %.172: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.cbc, %OrderedWith.facet.b02 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.b4e: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.b02) [concrete]
// CHECK:STDOUT:   %.680: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.b4e, %OrderedWith.facet.b02 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9d5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Ordered, %OrderedWith.facet.b02) [concrete]
// CHECK:STDOUT:   %.163: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9d5, %OrderedWith.facet.b02 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.8cd: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d91)]
// CHECK:STDOUT:   %Core.import_ref.010: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.2e2)]
// CHECK:STDOUT:   %Core.import_ref.16b: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.935)]
// CHECK:STDOUT:   %Core.import_ref.d00: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.dfc)]
// CHECK:STDOUT:   %Core.import_ref.a69: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1)]
// CHECK:STDOUT:   %Core.import_ref.ecd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.947541.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.bb34ec.1)]
// CHECK:STDOUT:   %Core.import_ref.ffd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.036982.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.ac2301.1)]
// CHECK:STDOUT:   %Core.import_ref.b46: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.104f17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.51e833.1)]
// CHECK:STDOUT:   %Core.import_ref.97f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.80a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fe3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1f1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc13(%U.loc13_17.2: type, %T.loc13_26.2: @OrderedWith.loc13.%OrderedWith.type.loc13_46.1 (%OrderedWith.type.507fad.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %OrderedWith.type.loc13_46.1 [symbolic = %require_complete.loc15 (constants.%require_complete.f6d)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.06a530.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.104f17.3)]
// CHECK:STDOUT:   %.loc15_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc13_26.1 [symbolic = %.loc15_5.2 (constants.%.236)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_26.1, @OrderedWith.1, @OrderedWith.1(%U.loc13_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc15_5.2: @OrderedWith.loc13.%.loc15_5.2 (%.236) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2: <specific function> = specific_impl_function %impl.elem0.loc15_5.2, @OrderedWith.WithSelf.Less(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.036982.3)]
// CHECK:STDOUT:   %.loc16_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc13_26.1 [symbolic = %.loc16_5.2 (constants.%.0de)]
// CHECK:STDOUT:   %impl.elem1.loc16_5.2: @OrderedWith.loc13.%.loc16_5.2 (%.0de) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2: <specific function> = specific_impl_function %impl.elem1.loc16_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.947541.3)]
// CHECK:STDOUT:   %.loc17_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc13_26.1 [symbolic = %.loc17_5.2 (constants.%.5e7)]
// CHECK:STDOUT:   %impl.elem2.loc17_5.2: @OrderedWith.loc13.%.loc17_5.2 (%.5e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2: <specific function> = specific_impl_function %impl.elem2.loc17_5.2, @OrderedWith.WithSelf.Greater(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3)]
// CHECK:STDOUT:   %.loc18_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc13_26.1 [symbolic = %.loc18_5.2 (constants.%.489)]
// CHECK:STDOUT:   %impl.elem3.loc18_5.2: @OrderedWith.loc13.%.loc18_5.2 (%.489) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2: <specific function> = specific_impl_function %impl.elem3.loc18_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc13_17.1, %T.loc13_26.1) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc13.%T.as_type.loc13_52.1 (%T.as_type), %y.param: @OrderedWith.loc13.%U.loc13_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc15: @OrderedWith.loc13.%T.as_type.loc13_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc15: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc15: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc15_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.8cd, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Less, %.loc15_5.1 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %impl.elem0.loc15_5.1: @OrderedWith.loc13.%.loc15_5.2 (%.236) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc15_5.1: <bound method> = bound_method %x.ref.loc15, %impl.elem0.loc15_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc15_5.1: <specific function> = specific_impl_function %impl.elem0.loc15_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc15_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:     %bound_method.loc15_5.2: <bound method> = bound_method %x.ref.loc15, %specific_impl_fn.loc15_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc15_5.2(%x.ref.loc15, %y.ref.loc15)
// CHECK:STDOUT:     %x.ref.loc16: @OrderedWith.loc13.%T.as_type.loc13_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc16: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc16: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc16_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.010, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref LessOrEquivalent, %.loc16_5.1 [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %impl.elem1.loc16_5.1: @OrderedWith.loc13.%.loc16_5.2 (%.0de) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc16_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc16_5.1: <bound method> = bound_method %x.ref.loc16, %impl.elem1.loc16_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_5.1: <specific function> = specific_impl_function %impl.elem1.loc16_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc16_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:     %bound_method.loc16_5.2: <bound method> = bound_method %x.ref.loc16, %specific_impl_fn.loc16_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc16_5.2(%x.ref.loc16, %y.ref.loc16)
// CHECK:STDOUT:     %x.ref.loc17: @OrderedWith.loc13.%T.as_type.loc13_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc17: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc17: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc17_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.16b, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Greater, %.loc17_5.1 [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %impl.elem2.loc17_5.1: @OrderedWith.loc13.%.loc17_5.2 (%.5e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc17_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc17_5.1: <bound method> = bound_method %x.ref.loc17, %impl.elem2.loc17_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc17_5.1: <specific function> = specific_impl_function %impl.elem2.loc17_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc17_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:     %bound_method.loc17_5.2: <bound method> = bound_method %x.ref.loc17, %specific_impl_fn.loc17_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc17_5.2(%x.ref.loc17, %y.ref.loc17)
// CHECK:STDOUT:     %x.ref.loc18: @OrderedWith.loc13.%T.as_type.loc13_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc18: @OrderedWith.loc13.%U.loc13_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc18: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc13_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc18_5.1: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.d00, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc13.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref GreaterOrEquivalent, %.loc18_5.1 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %impl.elem3.loc18_5.1: @OrderedWith.loc13.%.loc18_5.2 (%.489) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc18_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc18_5.1: <bound method> = bound_method %x.ref.loc18, %impl.elem3.loc18_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_5.1: <specific function> = specific_impl_function %impl.elem3.loc18_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc18_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:     %bound_method.loc18_5.2: <bound method> = bound_method %x.ref.loc18, %specific_impl_fn.loc18_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc18_5.2(%x.ref.loc18, %y.ref.loc18)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc13_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc13_46.1 => constants.%OrderedWith.type.507fad.2
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.88a
// CHECK:STDOUT:   %T.patt.loc13_26.2 => constants.%T.patt.b1d
// CHECK:STDOUT:   %T.loc13_26.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc13_52.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc13_50 => constants.%pattern_type.f17d83.2
// CHECK:STDOUT:   %x.param_patt.loc13_50.2 => constants.%x.param_patt.1a7
// CHECK:STDOUT:   %x.patt.loc13_50.2 => constants.%x.patt.708
// CHECK:STDOUT:   %pattern_type.loc13_56 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc13_56.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc13_56.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc13(constants.%Ordered, constants.%OrderedWith.facet.b02) {
// CHECK:STDOUT:   %U.patt.loc13_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%Ordered
// CHECK:STDOUT:   %OrderedWith.type.loc13_46.1 => constants.%OrderedWith.type.431
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.c99
// CHECK:STDOUT:   %T.patt.loc13_26.2 => constants.%T.patt.264
// CHECK:STDOUT:   %T.loc13_26.1 => constants.%OrderedWith.facet.b02
// CHECK:STDOUT:   %T.as_type.loc13_52.1 => constants.%Ordered
// CHECK:STDOUT:   %pattern_type.loc13_50 => constants.%pattern_type.e66
// CHECK:STDOUT:   %x.param_patt.loc13_50.2 => constants.%x.param_patt.e3d
// CHECK:STDOUT:   %x.patt.loc13_50.2 => constants.%x.patt.88c
// CHECK:STDOUT:   %pattern_type.loc13_56 => constants.%pattern_type.e66
// CHECK:STDOUT:   %y.param_patt.loc13_56.2 => constants.%y.param_patt.4a3
// CHECK:STDOUT:   %y.patt.loc13_56.2 => constants.%y.patt.68b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15 => constants.%complete_type.23a
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.e03
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c6f
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.ae1
// CHECK:STDOUT:   %.loc15_5.2 => constants.%.05c
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.c96
// CHECK:STDOUT:   %impl.elem0.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %specific_impl_fn.loc15_5.2 => constants.%Ordered.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.965
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.cbc
// CHECK:STDOUT:   %.loc16_5.2 => constants.%.172
// CHECK:STDOUT:   %impl.elem1.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc16_5.2 => constants.%Ordered.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.b19
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.b4e
// CHECK:STDOUT:   %.loc17_5.2 => constants.%.680
// CHECK:STDOUT:   %impl.elem2.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc17_5.2 => constants.%Ordered.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.673
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9d5
// CHECK:STDOUT:   %.loc18_5.2 => constants.%.163
// CHECK:STDOUT:   %impl.elem3.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc18_5.2 => constants.%Ordered.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_ordering.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.1: type = facet_type <@OrderedWith.1, @OrderedWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.635006.1: %OrderedWith.type.507fad.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.1: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1: %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.1: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.bb34ec.1: %OrderedWith.WithSelf.Greater.type.947541.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.ac2301.1: %OrderedWith.WithSelf.LessOrEquivalent.type.036982.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.1: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%Other, %Self.635006.1) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.51e833.1: %OrderedWith.WithSelf.Less.type.104f17.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %OrderedWith.type.507fad.2: type = facet_type <@OrderedWith.1, @OrderedWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.88a: type = pattern_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %T.patt.b1d: %pattern_type.88a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %OrderedWith.type.507fad.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.f17d83.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1a7: %pattern_type.f17d83.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.708: %pattern_type.f17d83.2 = wrapper_binding_pattern x, %x.param_patt.1a7 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.2 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %OrderedWith.assoc_type.06a530.2: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.10837a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic]
// CHECK:STDOUT:   %assoc1.4f4378.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic]
// CHECK:STDOUT:   %assoc2.60074a.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic]
// CHECK:STDOUT:   %assoc3.4e66f6.2: %OrderedWith.assoc_type.06a530.2 = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic]
// CHECK:STDOUT:   %require_complete.f6d: <witness> = require_complete_type %OrderedWith.type.507fad.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d91: %OrderedWith.assoc_type.06a530.1 = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @OrderedWith.1, @OrderedWith.1(%U) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.104f17.3: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.947541.3: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.236: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.104f17.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.236 = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.15a: <specific function> = specific_impl_function %impl.elem0, @OrderedWith.WithSelf.Less(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.2e2: %OrderedWith.assoc_type.06a530.1 = assoc_entity element1, imports.%Core.import_ref.80a [symbolic]
// CHECK:STDOUT:   %.0de: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.036982.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.0de = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1d5: <specific function> = specific_impl_function %impl.elem1, @OrderedWith.WithSelf.LessOrEquivalent(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc2.935: %OrderedWith.assoc_type.06a530.1 = assoc_entity element2, imports.%Core.import_ref.fe3 [symbolic]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.947541.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.5e7 = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.913: <specific function> = specific_impl_function %impl.elem2, @OrderedWith.WithSelf.Greater(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc3.dfc: %OrderedWith.assoc_type.06a530.1 = assoc_entity element3, imports.%Core.import_ref.1f1 [symbolic]
// CHECK:STDOUT:   %.489: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.489 = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e09: <specific function> = specific_impl_function %impl.elem3, @OrderedWith.WithSelf.GreaterOrEquivalent(%U, %T) [symbolic]
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
// CHECK:STDOUT:   %OrderedWith.type.988: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.b71: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.4bd: %OrderedWith.assoc_type.b71 = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.2dc: %OrderedWith.assoc_type.b71 = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.eda: %OrderedWith.assoc_type.b71 = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.c6a: %OrderedWith.assoc_type.b71 = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %A1.Less.type: type = fn_type @A1.Less [concrete]
// CHECK:STDOUT:   %A1.Less: %A1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent.type: type = fn_type @A1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.LessOrEquivalent: %A1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.Greater.type: type = fn_type @A1.Greater [concrete]
// CHECK:STDOUT:   %A1.Greater: %A1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent.type: type = fn_type @A1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %A1.GreaterOrEquivalent: %A1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6a8: <witness> = custom_witness (%A1.Less, %A1.LessOrEquivalent, %A1.Greater, %A1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.e28: %OrderedWith.type.988 = facet_value %A1, (%custom_witness.6a8) [concrete]
// CHECK:STDOUT:   %pattern_type.c0d: type = pattern_type %OrderedWith.type.988 [concrete]
// CHECK:STDOUT:   %T.patt.b5d: %pattern_type.c0d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.c7f: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.efd: %pattern_type.3b4 = wrapper_binding_pattern x, %x.param_patt.c7f [concrete]
// CHECK:STDOUT:   %y.param_patt.9d7: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.9fc: %pattern_type.5b2 = wrapper_binding_pattern y, %y.param_patt.9d7 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.716: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.808: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.6e7: %OrderedWith.assoc_type.808 = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.379: %OrderedWith.assoc_type.808 = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.cb3: %OrderedWith.assoc_type.808 = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.97d: %OrderedWith.assoc_type.808 = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %B1.Less.type: type = fn_type @B1.Less [concrete]
// CHECK:STDOUT:   %B1.Less: %B1.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent.type: type = fn_type @B1.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.LessOrEquivalent: %B1.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.Greater.type: type = fn_type @B1.Greater [concrete]
// CHECK:STDOUT:   %B1.Greater: %B1.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent.type: type = fn_type @B1.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B1.GreaterOrEquivalent: %B1.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.47d: <witness> = custom_witness (%B1.Less, %B1.LessOrEquivalent, %B1.Greater, %B1.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A1) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.fe6: %OrderedWith.type.716 = facet_value %B1, (%custom_witness.47d) [concrete]
// CHECK:STDOUT:   %pattern_type.850: type = pattern_type %OrderedWith.type.716 [concrete]
// CHECK:STDOUT:   %T.patt.16c: %pattern_type.850 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.bff: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.033: %pattern_type.5b2 = wrapper_binding_pattern x, %x.param_patt.bff [concrete]
// CHECK:STDOUT:   %y.param_patt.257: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.ab9: %pattern_type.3b4 = wrapper_binding_pattern y, %y.param_patt.257 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.b71: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.e02: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.24a: %OrderedWith.assoc_type.e02 = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.a62: %OrderedWith.assoc_type.e02 = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.aaf: %OrderedWith.assoc_type.e02 = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.56a: %OrderedWith.assoc_type.e02 = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %B2.Less.type: type = fn_type @B2.Less [concrete]
// CHECK:STDOUT:   %B2.Less: %B2.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent.type: type = fn_type @B2.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.LessOrEquivalent: %B2.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.Greater.type: type = fn_type @B2.Greater [concrete]
// CHECK:STDOUT:   %B2.Greater: %B2.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent.type: type = fn_type @B2.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B2.GreaterOrEquivalent: %B2.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fa3: <witness> = custom_witness (%B2.Less, %B2.LessOrEquivalent, %B2.Greater, %B2.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A2) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.a5b: %OrderedWith.type.b71 = facet_value %B2, (%custom_witness.fa3) [concrete]
// CHECK:STDOUT:   %pattern_type.eaf: type = pattern_type %OrderedWith.type.b71 [concrete]
// CHECK:STDOUT:   %T.patt.dc8: %pattern_type.eaf = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.0e6: %pattern_type.d1e = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.d0b: %pattern_type.d1e = wrapper_binding_pattern x, %x.param_patt.0e6 [concrete]
// CHECK:STDOUT:   %y.param_patt.a4f: %pattern_type.9f0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.a96: %pattern_type.9f0 = wrapper_binding_pattern y, %y.param_patt.a4f [concrete]
// CHECK:STDOUT:   %OrderedWith.type.dd5: type = facet_type <@OrderedWith.1, @OrderedWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.655: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.903: %OrderedWith.assoc_type.655 = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.b44: %OrderedWith.assoc_type.655 = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.b11: %OrderedWith.assoc_type.655 = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.39c: %OrderedWith.assoc_type.655 = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %A3.Less.type: type = fn_type @A3.Less [concrete]
// CHECK:STDOUT:   %A3.Less: %A3.Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %LessOrEquivalent.type: type = fn_type @LessOrEquivalent [concrete]
// CHECK:STDOUT:   %LessOrEquivalent: %LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %A3.Greater.type: type = fn_type @A3.Greater [concrete]
// CHECK:STDOUT:   %A3.Greater: %A3.Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent.type: type = fn_type @GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %GreaterOrEquivalent: %GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.411: <witness> = custom_witness (%A3.Less, %LessOrEquivalent, %A3.Greater, %GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%B3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.62b: %OrderedWith.type.dd5 = facet_value %A3, (%custom_witness.411) [concrete]
// CHECK:STDOUT:   %pattern_type.a14: type = pattern_type %OrderedWith.type.dd5 [concrete]
// CHECK:STDOUT:   %T.patt.87d: %pattern_type.a14 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.429: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.39e: %pattern_type.7fe = wrapper_binding_pattern x, %x.param_patt.429 [concrete]
// CHECK:STDOUT:   %y.param_patt.b51: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.e2b: %pattern_type.2dd = wrapper_binding_pattern y, %y.param_patt.b51 [concrete]
// CHECK:STDOUT:   %OrderedWith.type.649: type = facet_type <@OrderedWith.1, @OrderedWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %OrderedWith.assoc_type.4ba: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.f5b: %OrderedWith.assoc_type.4ba = assoc_entity element0, imports.%Core.import_ref.b46 [concrete]
// CHECK:STDOUT:   %assoc1.a3a: %OrderedWith.assoc_type.4ba = assoc_entity element1, imports.%Core.import_ref.ffd [concrete]
// CHECK:STDOUT:   %assoc2.e62: %OrderedWith.assoc_type.4ba = assoc_entity element2, imports.%Core.import_ref.ecd [concrete]
// CHECK:STDOUT:   %assoc3.18a: %OrderedWith.assoc_type.4ba = assoc_entity element3, imports.%Core.import_ref.a69 [concrete]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [concrete]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent.type: type = fn_type @B3.LessOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.LessOrEquivalent: %B3.LessOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent.type: type = fn_type @B3.GreaterOrEquivalent [concrete]
// CHECK:STDOUT:   %B3.GreaterOrEquivalent: %B3.GreaterOrEquivalent.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.545: <witness> = custom_witness (%Less, %B3.LessOrEquivalent, %Greater, %B3.GreaterOrEquivalent), @OrderedWith.1, @OrderedWith.1(%A3) [concrete]
// CHECK:STDOUT:   %OrderedWith.facet.432: %OrderedWith.type.649 = facet_value %B3, (%custom_witness.545) [concrete]
// CHECK:STDOUT:   %pattern_type.d98: type = pattern_type %OrderedWith.type.649 [concrete]
// CHECK:STDOUT:   %T.patt.742: %pattern_type.d98 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.074: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.9ae: %pattern_type.2dd = wrapper_binding_pattern x, %x.param_patt.074 [concrete]
// CHECK:STDOUT:   %y.param_patt.d86: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.426: %pattern_type.7fe = wrapper_binding_pattern y, %y.param_patt.d86 [concrete]
// CHECK:STDOUT:   %complete_type.dfc: <witness> = complete_type_witness %OrderedWith.type.988 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.1e6: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.e28) [concrete]
// CHECK:STDOUT:   %.a82: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.1e6, %OrderedWith.facet.e28 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.af3: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.e28) [concrete]
// CHECK:STDOUT:   %.fc5: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.af3, %OrderedWith.facet.e28 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.08d: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.e28) [concrete]
// CHECK:STDOUT:   %.b82: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.08d, %OrderedWith.facet.e28 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.9b6: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B1, %OrderedWith.facet.e28) [concrete]
// CHECK:STDOUT:   %.f3d: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.9b6, %OrderedWith.facet.e28 [concrete]
// CHECK:STDOUT:   %complete_type.79f: <witness> = complete_type_witness %OrderedWith.type.716 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.450: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.7bd: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.450, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.3d2: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.88c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.3d2, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.dea: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.2c7: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.dea, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.2ca: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A1, %OrderedWith.facet.fe6) [concrete]
// CHECK:STDOUT:   %.6cd: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.2ca, %OrderedWith.facet.fe6 [concrete]
// CHECK:STDOUT:   %complete_type.555: <witness> = complete_type_witness %OrderedWith.type.b71 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.5d6: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.a5b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.5d6, %OrderedWith.facet.a5b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.800: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.a5b) [concrete]
// CHECK:STDOUT:   %.27c: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.800, %OrderedWith.facet.a5b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.366: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.a5b) [concrete]
// CHECK:STDOUT:   %.229: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.366, %OrderedWith.facet.a5b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.31b: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A2, %OrderedWith.facet.a5b) [concrete]
// CHECK:STDOUT:   %.44b: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.31b, %OrderedWith.facet.a5b [concrete]
// CHECK:STDOUT:   %complete_type.977: <witness> = complete_type_witness %OrderedWith.type.dd5 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.5c7: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.62b) [concrete]
// CHECK:STDOUT:   %.49e: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.5c7, %OrderedWith.facet.62b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.db1: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.62b) [concrete]
// CHECK:STDOUT:   %.d0e: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.db1, %OrderedWith.facet.62b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.e53: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.62b) [concrete]
// CHECK:STDOUT:   %.523: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.e53, %OrderedWith.facet.62b [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.931: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%B3, %OrderedWith.facet.62b) [concrete]
// CHECK:STDOUT:   %.d38: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.931, %OrderedWith.facet.62b [concrete]
// CHECK:STDOUT:   %complete_type.f0f: <witness> = complete_type_witness %OrderedWith.type.649 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type.d08: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.432) [concrete]
// CHECK:STDOUT:   %.0c2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type.d08, %OrderedWith.facet.432 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type.6cd: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.432) [concrete]
// CHECK:STDOUT:   %.24d: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type.6cd, %OrderedWith.facet.432 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type.734: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.432) [concrete]
// CHECK:STDOUT:   %.589: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type.734, %OrderedWith.facet.432 [concrete]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type.4e5: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%A3, %OrderedWith.facet.432) [concrete]
// CHECK:STDOUT:   %.ed3: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type.4e5, %OrderedWith.facet.432 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.8cd: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc0 (constants.%assoc0.d91)]
// CHECK:STDOUT:   %Core.import_ref.010: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc1 (constants.%assoc1.2e2)]
// CHECK:STDOUT:   %Core.import_ref.16b: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc2 (constants.%assoc2.935)]
// CHECK:STDOUT:   %Core.import_ref.d00: @OrderedWith.WithSelf.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%assoc3 (constants.%assoc3.dfc)]
// CHECK:STDOUT:   %Core.import_ref.a69: @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent.type (%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.GreaterOrEquivalent (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.e10b91.1)]
// CHECK:STDOUT:   %Core.import_ref.ecd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater.type (%OrderedWith.WithSelf.Greater.type.947541.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Greater (constants.%OrderedWith.WithSelf.Greater.bb34ec.1)]
// CHECK:STDOUT:   %Core.import_ref.ffd: @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent.type (%OrderedWith.WithSelf.LessOrEquivalent.type.036982.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.LessOrEquivalent (constants.%OrderedWith.WithSelf.LessOrEquivalent.ac2301.1)]
// CHECK:STDOUT:   %Core.import_ref.b46: @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less.type (%OrderedWith.WithSelf.Less.type.104f17.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @OrderedWith.WithSelf.%OrderedWith.WithSelf.Less (constants.%OrderedWith.WithSelf.Less.51e833.1)]
// CHECK:STDOUT:   %Core.import_ref.97f = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.80a = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fe3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1f1 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @OrderedWith.loc54(%U.loc54_17.2: type, %T.loc54_26.2: @OrderedWith.loc54.%OrderedWith.type.loc54_46.1 (%OrderedWith.type.507fad.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc56: <witness> = require_complete_type %OrderedWith.type.loc54_46.1 [symbolic = %require_complete.loc56 (constants.%require_complete.f6d)]
// CHECK:STDOUT:   %OrderedWith.assoc_type: type = assoc_entity_type @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.assoc_type (constants.%OrderedWith.assoc_type.06a530.2)]
// CHECK:STDOUT:   %assoc0: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element0, imports.%Core.import_ref.b46 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type: type = fn_type @OrderedWith.WithSelf.Less, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %OrderedWith.WithSelf.Less.type (constants.%OrderedWith.WithSelf.Less.type.104f17.3)]
// CHECK:STDOUT:   %.loc56_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Less.type, %T.loc54_26.1 [symbolic = %.loc56_5.2 (constants.%.236)]
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc54_26.1, @OrderedWith.1, @OrderedWith.1(%U.loc54_17.1) [symbolic = %OrderedWith.lookup_impl_witness (constants.%OrderedWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc56_5.2: @OrderedWith.loc54.%.loc56_5.2 (%.236) = impl_witness_access %OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2: <specific function> = specific_impl_function %impl.elem0.loc56_5.2, @OrderedWith.WithSelf.Less(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:   %assoc1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element1, imports.%Core.import_ref.ffd [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.LessOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %OrderedWith.WithSelf.LessOrEquivalent.type (constants.%OrderedWith.WithSelf.LessOrEquivalent.type.036982.3)]
// CHECK:STDOUT:   %.loc57_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.LessOrEquivalent.type, %T.loc54_26.1 [symbolic = %.loc57_5.2 (constants.%.0de)]
// CHECK:STDOUT:   %impl.elem1.loc57_5.2: @OrderedWith.loc54.%.loc57_5.2 (%.0de) = impl_witness_access %OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2: <specific function> = specific_impl_function %impl.elem1.loc57_5.2, @OrderedWith.WithSelf.LessOrEquivalent(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:   %assoc2: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element2, imports.%Core.import_ref.ecd [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type: type = fn_type @OrderedWith.WithSelf.Greater, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %OrderedWith.WithSelf.Greater.type (constants.%OrderedWith.WithSelf.Greater.type.947541.3)]
// CHECK:STDOUT:   %.loc58_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.Greater.type, %T.loc54_26.1 [symbolic = %.loc58_5.2 (constants.%.5e7)]
// CHECK:STDOUT:   %impl.elem2.loc58_5.2: @OrderedWith.loc54.%.loc58_5.2 (%.5e7) = impl_witness_access %OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2: <specific function> = specific_impl_function %impl.elem2.loc58_5.2, @OrderedWith.WithSelf.Greater(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:   %assoc3: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = assoc_entity element3, imports.%Core.import_ref.a69 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type: type = fn_type @OrderedWith.WithSelf.GreaterOrEquivalent, @OrderedWith.WithSelf(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %OrderedWith.WithSelf.GreaterOrEquivalent.type (constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.fe291c.3)]
// CHECK:STDOUT:   %.loc59_5.2: type = fn_type_with_self_type %OrderedWith.WithSelf.GreaterOrEquivalent.type, %T.loc54_26.1 [symbolic = %.loc59_5.2 (constants.%.489)]
// CHECK:STDOUT:   %impl.elem3.loc59_5.2: @OrderedWith.loc54.%.loc59_5.2 (%.489) = impl_witness_access %OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2: <specific function> = specific_impl_function %impl.elem3.loc59_5.2, @OrderedWith.WithSelf.GreaterOrEquivalent(%U.loc54_17.1, %T.loc54_26.1) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @OrderedWith.loc54.%T.as_type.loc54_52.1 (%T.as_type), %y.param: @OrderedWith.loc54.%U.loc54_17.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc56: @OrderedWith.loc54.%T.as_type.loc54_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc56: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc56: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc56_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.8cd, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %Less.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Less, %.loc56_5.1 [symbolic = %assoc0 (constants.%assoc0.10837a.2)]
// CHECK:STDOUT:     %impl.elem0.loc56_5.1: @OrderedWith.loc54.%.loc56_5.2 (%.236) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc56_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc56_5.1: <bound method> = bound_method %x.ref.loc56, %impl.elem0.loc56_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc56_5.1: <specific function> = specific_impl_function %impl.elem0.loc56_5.1, @OrderedWith.WithSelf.Less(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc56_5.2 (constants.%specific_impl_fn.15a)]
// CHECK:STDOUT:     %bound_method.loc56_5.2: <bound method> = bound_method %x.ref.loc56, %specific_impl_fn.loc56_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Less.call: init bool = call %bound_method.loc56_5.2(%x.ref.loc56, %y.ref.loc56)
// CHECK:STDOUT:     %x.ref.loc57: @OrderedWith.loc54.%T.as_type.loc54_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc57: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc57: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc57_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.010, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %LessOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref LessOrEquivalent, %.loc57_5.1 [symbolic = %assoc1 (constants.%assoc1.4f4378.2)]
// CHECK:STDOUT:     %impl.elem1.loc57_5.1: @OrderedWith.loc54.%.loc57_5.2 (%.0de) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc57_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc57_5.1: <bound method> = bound_method %x.ref.loc57, %impl.elem1.loc57_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc57_5.1: <specific function> = specific_impl_function %impl.elem1.loc57_5.1, @OrderedWith.WithSelf.LessOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc57_5.2 (constants.%specific_impl_fn.1d5)]
// CHECK:STDOUT:     %bound_method.loc57_5.2: <bound method> = bound_method %x.ref.loc57, %specific_impl_fn.loc57_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.LessOrEquivalent.call: init bool = call %bound_method.loc57_5.2(%x.ref.loc57, %y.ref.loc57)
// CHECK:STDOUT:     %x.ref.loc58: @OrderedWith.loc54.%T.as_type.loc54_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc58: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc58: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc58_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.16b, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %Greater.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref Greater, %.loc58_5.1 [symbolic = %assoc2 (constants.%assoc2.60074a.2)]
// CHECK:STDOUT:     %impl.elem2.loc58_5.1: @OrderedWith.loc54.%.loc58_5.2 (%.5e7) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc58_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc58_5.1: <bound method> = bound_method %x.ref.loc58, %impl.elem2.loc58_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc58_5.1: <specific function> = specific_impl_function %impl.elem2.loc58_5.1, @OrderedWith.WithSelf.Greater(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc58_5.2 (constants.%specific_impl_fn.913)]
// CHECK:STDOUT:     %bound_method.loc58_5.2: <bound method> = bound_method %x.ref.loc58, %specific_impl_fn.loc58_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.Greater.call: init bool = call %bound_method.loc58_5.2(%x.ref.loc58, %y.ref.loc58)
// CHECK:STDOUT:     %x.ref.loc59: @OrderedWith.loc54.%T.as_type.loc54_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc59: @OrderedWith.loc54.%U.loc54_17.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %OrderedWith.type.loc59: type = facet_type <@OrderedWith.1, @OrderedWith.1(constants.%U)> [symbolic = %OrderedWith.type.loc54_46.1 (constants.%OrderedWith.type.507fad.2)]
// CHECK:STDOUT:     %.loc59_5.1: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = specific_constant imports.%Core.import_ref.d00, @OrderedWith.WithSelf(constants.%U, constants.%Self.635006.1) [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %GreaterOrEquivalent.ref: @OrderedWith.loc54.%OrderedWith.assoc_type (%OrderedWith.assoc_type.06a530.2) = name_ref GreaterOrEquivalent, %.loc59_5.1 [symbolic = %assoc3 (constants.%assoc3.4e66f6.2)]
// CHECK:STDOUT:     %impl.elem3.loc59_5.1: @OrderedWith.loc54.%.loc59_5.2 (%.489) = impl_witness_access constants.%OrderedWith.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc59_5.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc59_5.1: <bound method> = bound_method %x.ref.loc59, %impl.elem3.loc59_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc59_5.1: <specific function> = specific_impl_function %impl.elem3.loc59_5.1, @OrderedWith.WithSelf.GreaterOrEquivalent(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc59_5.2 (constants.%specific_impl_fn.e09)]
// CHECK:STDOUT:     %bound_method.loc59_5.2: <bound method> = bound_method %x.ref.loc59, %specific_impl_fn.loc59_5.1
// CHECK:STDOUT:     %OrderedWith.WithSelf.GreaterOrEquivalent.call: init bool = call %bound_method.loc59_5.2(%x.ref.loc59, %y.ref.loc59)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%U
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.507fad.2
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.88a
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.b1d
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.f17d83.2
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.1a7
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.708
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B1, constants.%OrderedWith.facet.e28) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B1
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.988
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.c0d
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.b5d
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%OrderedWith.facet.e28
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.c7f
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.efd
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.9d7
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.9fc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.dfc
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.b71
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4bd
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.1e6
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.a82
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.6a8
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2dc
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.af3
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.fc5
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%A1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.eda
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.08d
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.b82
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.c6a
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.9b6
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.f3d
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%A1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A1, constants.%OrderedWith.facet.fe6) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A1
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.716
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.850
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.16c
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%OrderedWith.facet.fe6
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.bff
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.033
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.257
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.ab9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.79f
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.808
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6e7
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.450
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.7bd
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.47d
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B1.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.379
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.3d2
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.88c
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B1.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.cb3
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.dea
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.2c7
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B1.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.97d
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.2ca
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.6cd
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B1.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A2, constants.%OrderedWith.facet.a5b) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A2
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.b71
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.eaf
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.dc8
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%OrderedWith.facet.a5b
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.d1e
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.0e6
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.d0b
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.9f0
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.a4f
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.a96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.555
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.e02
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.24a
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.5d6
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.3c4
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.fa3
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%B2.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a62
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.800
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.27c
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B2.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.aaf
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.366
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.229
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%B2.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.56a
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.31b
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.44b
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B2.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%B3, constants.%OrderedWith.facet.62b) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%B3
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.dd5
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.a14
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.87d
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%OrderedWith.facet.62b
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.429
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.39e
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.b51
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.e2b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.977
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.655
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.903
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.5c7
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.49e
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.411
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%A3.Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b44
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.db1
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.d0e
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.b11
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.e53
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.523
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%A3.Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.39c
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.931
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.d38
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @OrderedWith.loc54(constants.%A3, constants.%OrderedWith.facet.432) {
// CHECK:STDOUT:   %U.patt.loc54_17.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc54_17.1 => constants.%A3
// CHECK:STDOUT:   %OrderedWith.type.loc54_46.1 => constants.%OrderedWith.type.649
// CHECK:STDOUT:   %pattern_type.loc54_26 => constants.%pattern_type.d98
// CHECK:STDOUT:   %T.patt.loc54_26.2 => constants.%T.patt.742
// CHECK:STDOUT:   %T.loc54_26.1 => constants.%OrderedWith.facet.432
// CHECK:STDOUT:   %T.as_type.loc54_52.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc54_50 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %x.param_patt.loc54_50.2 => constants.%x.param_patt.074
// CHECK:STDOUT:   %x.patt.loc54_50.2 => constants.%x.patt.9ae
// CHECK:STDOUT:   %pattern_type.loc54_56 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %y.param_patt.loc54_56.2 => constants.%y.param_patt.d86
// CHECK:STDOUT:   %y.patt.loc54_56.2 => constants.%y.patt.426
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc54_50 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc54_56 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.f0f
// CHECK:STDOUT:   %OrderedWith.assoc_type => constants.%OrderedWith.assoc_type.4ba
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f5b
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.type => constants.%OrderedWith.WithSelf.Less.type.d08
// CHECK:STDOUT:   %.loc56_5.2 => constants.%.0c2
// CHECK:STDOUT:   %OrderedWith.lookup_impl_witness => constants.%custom_witness.545
// CHECK:STDOUT:   %impl.elem0.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %specific_impl_fn.loc56_5.2 => constants.%Less
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a3a
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.type => constants.%OrderedWith.WithSelf.LessOrEquivalent.type.6cd
// CHECK:STDOUT:   %.loc57_5.2 => constants.%.24d
// CHECK:STDOUT:   %impl.elem1.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc57_5.2 => constants.%B3.LessOrEquivalent
// CHECK:STDOUT:   %assoc2 => constants.%assoc2.e62
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.type => constants.%OrderedWith.WithSelf.Greater.type.734
// CHECK:STDOUT:   %.loc58_5.2 => constants.%.589
// CHECK:STDOUT:   %impl.elem2.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %specific_impl_fn.loc58_5.2 => constants.%Greater
// CHECK:STDOUT:   %assoc3 => constants.%assoc3.18a
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.type => constants.%OrderedWith.WithSelf.GreaterOrEquivalent.type.4e5
// CHECK:STDOUT:   %.loc59_5.2 => constants.%.ed3
// CHECK:STDOUT:   %impl.elem3.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT:   %specific_impl_fn.loc59_5.2 => constants.%B3.GreaterOrEquivalent
// CHECK:STDOUT: }
// CHECK:STDOUT:
