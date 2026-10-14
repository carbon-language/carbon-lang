// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/eq_with.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/eq_with.carbon

// --- returns_bool.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct EqualityComparable {
  auto operator==(EqualityComparable const&) const -> bool;
  auto operator!=(EqualityComparable const&) const -> bool;
};
''';

fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x == y;
  x != y;
  //@dump-sem-ir-end
}

fn Test(equality_comparable: Cpp.EqualityComparable) {
  EqWith(equality_comparable, equality_comparable);
}

// --- cross_type_equality.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct B1;
struct A1 {
  auto operator==(B1) const -> bool;
  auto operator!=(B1) const -> bool;
};
struct B1 {
  auto operator==(A1) const -> bool;
  auto operator!=(A1) const -> bool;
};

struct B2;
struct A2 {
};
struct B2 {
  auto operator==(A2) const -> bool;
  auto operator!=(A2) const -> bool;
};

struct B3;
struct A3 {
  auto operator==(B3) const -> bool;
  friend auto operator==(B3, A3) -> bool;
};
struct B3 {
  auto operator!=(A3) const -> bool;
  friend auto operator!=(A3, B3) -> bool;
};
''';

fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  //@dump-sem-ir-begin
  x == y;
  x != y;
  //@dump-sem-ir-end
}

fn Test(a1: Cpp.A1, b1: Cpp.B1,
        a2: Cpp.A2, b2: Cpp.B2,
        a3: Cpp.A3, b3: Cpp.B3) {
  EqWith(a1, b1);
  EqWith(b1, a1);

  EqWith(b2, a2);

  EqWith(a3, b3);
  EqWith(b3, a3);
}

// --- fail_todo_returns_boolish.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct NeqBoolish {
  auto operator==(NeqBoolish) const -> bool;
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+9]]:8: error: cannot implicitly convert expression of type `i32` to `bool` [ConversionFailure]
  // CHECK:STDERR:   auto operator!=(NeqBoolish) const -> int;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+6]]:8: note: type `i32` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   auto operator!=(NeqBoolish) const -> int;
  // CHECK:STDERR:        ^
  // CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:14:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   default fn NotEqual(self, other: Other) -> bool {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  auto operator!=(NeqBoolish) const -> int;
};

struct EqBoolish {
  auto operator==(EqBoolish) const -> int*;
  auto operator!=(EqBoolish) const -> bool;
};

struct BothBoolish {
  enum Result {
    False,
    True,
  };
  auto operator==(BothBoolish) const -> Result;
  auto operator!=(BothBoolish) const -> Result;
};
''';

// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-18]]:8: error: cannot implicitly convert expression of type `Core.Optional(i32* as Core.OptionalStorage)` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator==(EqBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-21]]:8: note: type `Core.Optional(i32* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator==(EqBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:13:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Equal(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+30]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-22]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator==(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-25]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator==(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:13:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Equal(self, other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-34]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator!=(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-37]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator!=(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:14:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   default fn NotEqual(self, other: Other) -> bool {
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(neq_boolish: Cpp.NeqBoolish,
        eq_boolish: Cpp.EqBoolish,
        both_boolish: Cpp.BothBoolish) {
  EqWith(neq_boolish, neq_boolish);

  EqWith(eq_boolish, eq_boolish);

  EqWith(both_boolish, both_boolish);
}

// --- fail_todo_comparison_rewrites.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct DefaultEqOnly {
public:
  auto operator==(const DefaultEqOnly&) const -> bool = default;
};

struct OnlyEqDeclared {
public:
  auto operator==(OnlyEqDeclared) const -> bool;
};
''';

fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(default_eq_only: Cpp.DefaultEqOnly,
        // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+4]]:33: error: member name `DefaultSpaceshipOnly` not found in `Cpp` [MemberNameNotFoundInInstScope]
        // CHECK:STDERR:         default_spaceship_only: Cpp.DefaultSpaceshipOnly,
        // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~~~~
        // CHECK:STDERR:
        default_spaceship_only: Cpp.DefaultSpaceshipOnly,
        only_eq_declared: Cpp.OnlyEqDeclared) {
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(default_eq_only, default_eq_only);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(default_eq_only, default_eq_only);

  EqWith(default_spaceship_only, default_spaceship_only);

  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(only_eq_declared, only_eq_declared);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-26]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(only_eq_declared, only_eq_declared);
}

// --- fail_never_valid.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct OnlyNeq {
  auto operator!=(OnlyNeq) const -> bool;
};

struct NoRelevantMembers {};

struct EqReturnsVoid {
  auto operator==(EqReturnsVoid) const -> void;
  auto operator!=(EqReturnsVoid) const -> bool;
};

struct NeqReturnsVoid {
  auto operator==(NeqReturnsVoid) const -> bool;
  auto operator!=(NeqReturnsVoid) const -> void;
};

struct DeletedNeq {
  auto operator==(DeletedNeq) const -> bool;
  auto operator!=(DeletedNeq) const -> bool = delete;
};

struct NonConstEq {
  auto operator==(NonConstEq) -> bool;
  auto operator!=(NonConstEq) const -> bool;
};

struct A2 {};
struct B2 {
  auto operator==(A2) const -> bool;
  auto operator!=(A2) const -> bool;
};
''';

fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Test(only_neq: Cpp.OnlyNeq,
        no_relevant_members: Cpp.NoRelevantMembers,
        eq_returns_void: Cpp.EqReturnsVoid,
        neq_returns_void: Cpp.NeqReturnsVoid,
        deleted_neq: Cpp.DeletedNeq,
        non_const_eq: Cpp.NonConstEq,
        a2: Cpp.A2, b2: Cpp.B2) {
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.OnlyNeq` into type implementing `Core.EqWith(Cpp.OnlyNeq)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(only_neq, only_neq);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(only_neq, only_neq);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+8]]:10: error: name `only_non_default_spaceship` not found [NameNotFound]
  // CHECK:STDERR:   EqWith(only_non_default_spaceship, only_non_default_spaceship);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+4]]:38: error: name `only_non_default_spaceship` not found [NameNotFound]
  // CHECK:STDERR:   EqWith(only_non_default_spaceship, only_non_default_spaceship);
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(only_non_default_spaceship, only_non_default_spaceship);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoRelevantMembers` into type implementing `Core.EqWith(Cpp.NoRelevantMembers)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(no_relevant_members, no_relevant_members);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-34]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(no_relevant_members, no_relevant_members);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EqReturnsVoid` into type implementing `Core.EqWith(Cpp.EqReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(eq_returns_void, eq_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(eq_returns_void, eq_returns_void);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NeqReturnsVoid` into type implementing `Core.EqWith(Cpp.NeqReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(neq_returns_void, neq_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(neq_returns_void, neq_returns_void);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+13]]:34: error: overload resolution selected deleted operator '!=' [CppInteropParseError]
  // CHECK:STDERR:   109 |   EqWith(deleted_neq, deleted_neq);
  // CHECK:STDERR:       |                                  ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-77]]:8: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:    22 |   auto operator==(DeletedNeq) const -> bool;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-79]]:8: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    23 |   auto operator!=(DeletedNeq) const -> bool = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-67]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(deleted_neq, deleted_neq);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(non_const_eq, non_const_eq);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-76]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(non_const_eq, non_const_eq);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.A2` into type implementing `Core.EqWith(Cpp.B2)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(a2, b2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-85]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(a2, b2);
}

// CHECK:STDOUT: --- returns_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bd18d5.1: %EqWith.type.686248.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.cb1497.1: %EqWith.WithSelf.NotEqual.type.0896a8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.74d211.1: %EqWith.WithSelf.Equal.type.4e5264.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.295: type = pattern_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %T.patt.738: %pattern_type.295 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.686248.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.d13aea.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1bf: %pattern_type.d13aea.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.c79: %pattern_type.d13aea.2 = wrapper_binding_pattern x, %x.param_patt.1bf [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.2 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.5a05b6.2: %EqWith.assoc_type.92e927.2 = assoc_entity element0, imports.%Core.import_ref.def [symbolic]
// CHECK:STDOUT:   %assoc1.09a898.2: %EqWith.assoc_type.92e927.2 = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic]
// CHECK:STDOUT:   %require_complete.8a6cdd.2: <witness> = require_complete_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %assoc0.485f43.3: %EqWith.assoc_type.92e927.1 = assoc_entity element0, imports.%Core.import_ref.a05e82.3 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness.ae2c3a.2: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.161b4c.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.4e5264.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.52b4ea.2: %.161b4c.2 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2d3bf0.2: <specific function> = specific_impl_function %impl.elem0.52b4ea.2, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.0bbec6.2: %EqWith.assoc_type.92e927.1 = assoc_entity element1, imports.%Core.import_ref.7aa136.2 [symbolic]
// CHECK:STDOUT:   %.814: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0896a8.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.814 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0b5: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqualityComparable: type = class_type @EqualityComparable [concrete]
// CHECK:STDOUT:   %pattern_type.73b: type = pattern_type %EqualityComparable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EqWith.type.d0d: type = facet_type <@EqWith.1, @EqWith.1(%EqualityComparable)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.fed: type = assoc_entity_type @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %assoc0.cc6: %EqWith.assoc_type.fed = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.df5: %EqWith.assoc_type.fed = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal.type: type = fn_type @EqualityComparable.Equal [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal: %EqualityComparable.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual.type: type = fn_type @EqualityComparable.NotEqual [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual: %EqualityComparable.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5bf: <witness> = custom_witness (%EqualityComparable.Equal, %EqualityComparable.NotEqual), @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %EqWith.facet.fd4: %EqWith.type.d0d = facet_value %EqualityComparable, (%custom_witness.5bf) [concrete]
// CHECK:STDOUT:   %pattern_type.90a: type = pattern_type %EqWith.type.d0d [concrete]
// CHECK:STDOUT:   %T.patt.d4b: %pattern_type.90a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.427: %pattern_type.73b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e43: %pattern_type.73b = wrapper_binding_pattern x, %x.param_patt.427 [concrete]
// CHECK:STDOUT:   %y.param_patt.464: %pattern_type.73b = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.522: %pattern_type.73b = wrapper_binding_pattern y, %y.param_patt.464 [concrete]
// CHECK:STDOUT:   %complete_type.7c4: <witness> = complete_type_witness %EqWith.type.d0d [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.689: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.fd4) [concrete]
// CHECK:STDOUT:   %.4d9: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.689, %EqWith.facet.fd4 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.42a: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.fd4) [concrete]
// CHECK:STDOUT:   %.dc0: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.42a, %EqWith.facet.fd4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f86: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.485f43.3)]
// CHECK:STDOUT:   %Core.import_ref.91b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.0bbec6.2)]
// CHECK:STDOUT:   %Core.import_ref.6f1: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.0896a8.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.cb1497.1)]
// CHECK:STDOUT:   %Core.import_ref.def: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.4e5264.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.74d211.1)]
// CHECK:STDOUT:   %Core.import_ref.a05e82.3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7aa136.2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc11(%U.loc11_12.2: type, %T.loc11_21.2: @EqWith.loc11.%EqWith.type.loc11_36.1 (%EqWith.type.686248.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %EqWith.type.loc11_36.1 [symbolic = %require_complete.loc13 (constants.%require_complete.8a6cdd.2)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.92e927.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element0, imports.%Core.import_ref.def [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_21.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.4e5264.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc11_21.1 [symbolic = %.loc13_5.2 (constants.%.161b4c.2)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_21.1, @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness.ae2c3a.2)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @EqWith.loc11.%.loc13_5.2 (%.161b4c.2) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @EqWith.WithSelf.Equal(%U.loc11_12.1, %T.loc11_21.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_21.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.0896a8.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc11_21.1 [symbolic = %.loc14_5.2 (constants.%.814)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @EqWith.loc11.%.loc14_5.2 (%.814) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @EqWith.WithSelf.NotEqual(%U.loc11_12.1, %T.loc11_21.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc11.%T.as_type.loc11_42.1 (%T.as_type), %y.param: @EqWith.loc11.%U.loc11_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @EqWith.loc11.%T.as_type.loc11_42.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc13: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc13_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.f86, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref Equal, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @EqWith.loc11.%.loc13_5.2 (%.161b4c.2) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @EqWith.loc11.%T.as_type.loc11_42.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc14: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc14_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.91b, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref NotEqual, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @EqWith.loc11.%.loc14_5.2 (%.814) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc11_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc11_36.1 => constants.%EqWith.type.686248.2
// CHECK:STDOUT:   %pattern_type.loc11_21 => constants.%pattern_type.295
// CHECK:STDOUT:   %T.patt.loc11_21.2 => constants.%T.patt.738
// CHECK:STDOUT:   %T.loc11_21.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc11_42.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc11_40 => constants.%pattern_type.d13aea.2
// CHECK:STDOUT:   %x.param_patt.loc11_40.2 => constants.%x.param_patt.1bf
// CHECK:STDOUT:   %x.patt.loc11_40.2 => constants.%x.patt.c79
// CHECK:STDOUT:   %pattern_type.loc11_46 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc11_46.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc11_46.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%EqualityComparable, constants.%EqWith.facet.fd4) {
// CHECK:STDOUT:   %U.patt.loc11_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %EqWith.type.loc11_36.1 => constants.%EqWith.type.d0d
// CHECK:STDOUT:   %pattern_type.loc11_21 => constants.%pattern_type.90a
// CHECK:STDOUT:   %T.patt.loc11_21.2 => constants.%T.patt.d4b
// CHECK:STDOUT:   %T.loc11_21.1 => constants.%EqWith.facet.fd4
// CHECK:STDOUT:   %T.as_type.loc11_42.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %pattern_type.loc11_40 => constants.%pattern_type.73b
// CHECK:STDOUT:   %x.param_patt.loc11_40.2 => constants.%x.param_patt.427
// CHECK:STDOUT:   %x.patt.loc11_40.2 => constants.%x.patt.e43
// CHECK:STDOUT:   %pattern_type.loc11_46 => constants.%pattern_type.73b
// CHECK:STDOUT:   %y.param_patt.loc11_46.2 => constants.%y.param_patt.464
// CHECK:STDOUT:   %y.patt.loc11_46.2 => constants.%y.patt.522
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13 => constants.%complete_type.7c4
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.fed
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cc6
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.689
// CHECK:STDOUT:   %.loc13_5.2 => constants.%.4d9
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.5bf
// CHECK:STDOUT:   %impl.elem0.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.df5
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.42a
// CHECK:STDOUT:   %.loc14_5.2 => constants.%.dc0
// CHECK:STDOUT:   %impl.elem1.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_equality.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bd18d5.1: %EqWith.type.686248.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.cb1497.1: %EqWith.WithSelf.NotEqual.type.0896a8.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.bd18d5.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.74d211.1: %EqWith.WithSelf.Equal.type.4e5264.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.686248.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.295: type = pattern_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %T.patt.738: %pattern_type.295 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.686248.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.d13aea.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1bf: %pattern_type.d13aea.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.c79: %pattern_type.d13aea.2 = wrapper_binding_pattern x, %x.param_patt.1bf [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.2 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.92e927.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.5a05b6.2: %EqWith.assoc_type.92e927.2 = assoc_entity element0, imports.%Core.import_ref.def [symbolic]
// CHECK:STDOUT:   %assoc1.09a898.2: %EqWith.assoc_type.92e927.2 = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic]
// CHECK:STDOUT:   %require_complete.8a6cdd.2: <witness> = require_complete_type %EqWith.type.686248.2 [symbolic]
// CHECK:STDOUT:   %assoc0.485f43.3: %EqWith.assoc_type.92e927.1 = assoc_entity element0, imports.%Core.import_ref.a05e82.3 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness.ae2c3a.2: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e5264.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0896a8.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.161b4c.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.4e5264.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0.52b4ea.2: %.161b4c.2 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2d3bf0.2: <specific function> = specific_impl_function %impl.elem0.52b4ea.2, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.0bbec6.2: %EqWith.assoc_type.92e927.1 = assoc_entity element1, imports.%Core.import_ref.7aa136.2 [symbolic]
// CHECK:STDOUT:   %.814: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0896a8.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.814 = impl_witness_access %EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0b5: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
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
// CHECK:STDOUT:   %EqWith.type.10c: type = facet_type <@EqWith.1, @EqWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.d5f: type = assoc_entity_type @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.e9c: %EqWith.assoc_type.d5f = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.39e: %EqWith.assoc_type.d5f = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %A1.Equal.type: type = fn_type @A1.Equal [concrete]
// CHECK:STDOUT:   %A1.Equal: %A1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.NotEqual.type: type = fn_type @A1.NotEqual [concrete]
// CHECK:STDOUT:   %A1.NotEqual: %A1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.908: <witness> = custom_witness (%A1.Equal, %A1.NotEqual), @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.c0f: %EqWith.type.10c = facet_value %A1, (%custom_witness.908) [concrete]
// CHECK:STDOUT:   %pattern_type.e52: type = pattern_type %EqWith.type.10c [concrete]
// CHECK:STDOUT:   %T.patt.d5d: %pattern_type.e52 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.c7f: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.efd: %pattern_type.3b4 = wrapper_binding_pattern x, %x.param_patt.c7f [concrete]
// CHECK:STDOUT:   %y.param_patt.9d7: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.9fc: %pattern_type.5b2 = wrapper_binding_pattern y, %y.param_patt.9d7 [concrete]
// CHECK:STDOUT:   %EqWith.type.49a: type = facet_type <@EqWith.1, @EqWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.e22: type = assoc_entity_type @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.ac3: %EqWith.assoc_type.e22 = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.8af: %EqWith.assoc_type.e22 = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %B1.Equal.type: type = fn_type @B1.Equal [concrete]
// CHECK:STDOUT:   %B1.Equal: %B1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.NotEqual.type: type = fn_type @B1.NotEqual [concrete]
// CHECK:STDOUT:   %B1.NotEqual: %B1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.abd: <witness> = custom_witness (%B1.Equal, %B1.NotEqual), @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.56f: %EqWith.type.49a = facet_value %B1, (%custom_witness.abd) [concrete]
// CHECK:STDOUT:   %pattern_type.86e: type = pattern_type %EqWith.type.49a [concrete]
// CHECK:STDOUT:   %T.patt.4c6: %pattern_type.86e = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.bff: %pattern_type.5b2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.033: %pattern_type.5b2 = wrapper_binding_pattern x, %x.param_patt.bff [concrete]
// CHECK:STDOUT:   %y.param_patt.257: %pattern_type.3b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.ab9: %pattern_type.3b4 = wrapper_binding_pattern y, %y.param_patt.257 [concrete]
// CHECK:STDOUT:   %EqWith.type.6a6: type = facet_type <@EqWith.1, @EqWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.491: type = assoc_entity_type @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.341: %EqWith.assoc_type.491 = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.0c5: %EqWith.assoc_type.491 = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %B2.Equal.type: type = fn_type @B2.Equal [concrete]
// CHECK:STDOUT:   %B2.Equal: %B2.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.NotEqual.type: type = fn_type @B2.NotEqual [concrete]
// CHECK:STDOUT:   %B2.NotEqual: %B2.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b31: <witness> = custom_witness (%B2.Equal, %B2.NotEqual), @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %EqWith.facet.1c6: %EqWith.type.6a6 = facet_value %B2, (%custom_witness.b31) [concrete]
// CHECK:STDOUT:   %pattern_type.8fb: type = pattern_type %EqWith.type.6a6 [concrete]
// CHECK:STDOUT:   %T.patt.0c0: %pattern_type.8fb = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.0e6: %pattern_type.d1e = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.d0b: %pattern_type.d1e = wrapper_binding_pattern x, %x.param_patt.0e6 [concrete]
// CHECK:STDOUT:   %y.param_patt.a4f: %pattern_type.9f0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.a96: %pattern_type.9f0 = wrapper_binding_pattern y, %y.param_patt.a4f [concrete]
// CHECK:STDOUT:   %EqWith.type.131: type = facet_type <@EqWith.1, @EqWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.a72: type = assoc_entity_type @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.756: %EqWith.assoc_type.a72 = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.0c8: %EqWith.assoc_type.a72 = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %A3.Equal.type: type = fn_type @A3.Equal [concrete]
// CHECK:STDOUT:   %A3.Equal: %A3.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.910: <witness> = custom_witness (%A3.Equal, %NotEqual), @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.c7d: %EqWith.type.131 = facet_value %A3, (%custom_witness.910) [concrete]
// CHECK:STDOUT:   %pattern_type.b8d: type = pattern_type %EqWith.type.131 [concrete]
// CHECK:STDOUT:   %T.patt.135: %pattern_type.b8d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.429: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.39e: %pattern_type.7fe = wrapper_binding_pattern x, %x.param_patt.429 [concrete]
// CHECK:STDOUT:   %y.param_patt.b51: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.e2b: %pattern_type.2dd = wrapper_binding_pattern y, %y.param_patt.b51 [concrete]
// CHECK:STDOUT:   %EqWith.type.386: type = facet_type <@EqWith.1, @EqWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.532: type = assoc_entity_type @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.e81: %EqWith.assoc_type.532 = assoc_entity element0, imports.%Core.import_ref.def [concrete]
// CHECK:STDOUT:   %assoc1.a50: %EqWith.assoc_type.532 = assoc_entity element1, imports.%Core.import_ref.6f1 [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.NotEqual.type: type = fn_type @B3.NotEqual [concrete]
// CHECK:STDOUT:   %B3.NotEqual: %B3.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7d4: <witness> = custom_witness (%Equal, %B3.NotEqual), @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.43a: %EqWith.type.386 = facet_value %B3, (%custom_witness.7d4) [concrete]
// CHECK:STDOUT:   %pattern_type.9d5: type = pattern_type %EqWith.type.386 [concrete]
// CHECK:STDOUT:   %T.patt.352: %pattern_type.9d5 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.074: %pattern_type.2dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.9ae: %pattern_type.2dd = wrapper_binding_pattern x, %x.param_patt.074 [concrete]
// CHECK:STDOUT:   %y.param_patt.d86: %pattern_type.7fe = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.426: %pattern_type.7fe = wrapper_binding_pattern y, %y.param_patt.d86 [concrete]
// CHECK:STDOUT:   %complete_type.1e8: <witness> = complete_type_witness %EqWith.type.10c [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b39: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B1, %EqWith.facet.c0f) [concrete]
// CHECK:STDOUT:   %.a54: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b39, %EqWith.facet.c0f [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.7ce: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B1, %EqWith.facet.c0f) [concrete]
// CHECK:STDOUT:   %.c68: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.7ce, %EqWith.facet.c0f [concrete]
// CHECK:STDOUT:   %complete_type.2d1: <witness> = complete_type_witness %EqWith.type.49a [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.872: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A1, %EqWith.facet.56f) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.872, %EqWith.facet.56f [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.bdb: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A1, %EqWith.facet.56f) [concrete]
// CHECK:STDOUT:   %.15a: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.bdb, %EqWith.facet.56f [concrete]
// CHECK:STDOUT:   %complete_type.5d1: <witness> = complete_type_witness %EqWith.type.6a6 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.681: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A2, %EqWith.facet.1c6) [concrete]
// CHECK:STDOUT:   %.97f: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.681, %EqWith.facet.1c6 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.616: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A2, %EqWith.facet.1c6) [concrete]
// CHECK:STDOUT:   %.6e3: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.616, %EqWith.facet.1c6 [concrete]
// CHECK:STDOUT:   %complete_type.7da: <witness> = complete_type_witness %EqWith.type.131 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.a47: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B3, %EqWith.facet.c7d) [concrete]
// CHECK:STDOUT:   %.efa: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.a47, %EqWith.facet.c7d [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.577: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B3, %EqWith.facet.c7d) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.577, %EqWith.facet.c7d [concrete]
// CHECK:STDOUT:   %complete_type.05b: <witness> = complete_type_witness %EqWith.type.386 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.4e8: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A3, %EqWith.facet.43a) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.4e8, %EqWith.facet.43a [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.0c8: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A3, %EqWith.facet.43a) [concrete]
// CHECK:STDOUT:   %.82c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.0c8, %EqWith.facet.43a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.f86: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.485f43.3)]
// CHECK:STDOUT:   %Core.import_ref.91b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.92e927.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.0bbec6.2)]
// CHECK:STDOUT:   %Core.import_ref.6f1: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.0896a8.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.cb1497.1)]
// CHECK:STDOUT:   %Core.import_ref.def: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.4e5264.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.74d211.1)]
// CHECK:STDOUT:   %Core.import_ref.a05e82.3 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7aa136.2 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc34(%U.loc34_12.2: type, %T.loc34_21.2: @EqWith.loc34.%EqWith.type.loc34_36.1 (%EqWith.type.686248.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc36: <witness> = require_complete_type %EqWith.type.loc34_36.1 [symbolic = %require_complete.loc36 (constants.%require_complete.8a6cdd.2)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.92e927.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element0, imports.%Core.import_ref.def [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_21.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.4e5264.3)]
// CHECK:STDOUT:   %.loc36_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc34_21.1 [symbolic = %.loc36_5.2 (constants.%.161b4c.2)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc34_21.1, @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness.ae2c3a.2)]
// CHECK:STDOUT:   %impl.elem0.loc36_5.2: @EqWith.loc34.%.loc36_5.2 (%.161b4c.2) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2: <specific function> = specific_impl_function %impl.elem0.loc36_5.2, @EqWith.WithSelf.Equal(%U.loc34_12.1, %T.loc34_21.1) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = assoc_entity element1, imports.%Core.import_ref.6f1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_21.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.0896a8.3)]
// CHECK:STDOUT:   %.loc37_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc34_21.1 [symbolic = %.loc37_5.2 (constants.%.814)]
// CHECK:STDOUT:   %impl.elem1.loc37_5.2: @EqWith.loc34.%.loc37_5.2 (%.814) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2: <specific function> = specific_impl_function %impl.elem1.loc37_5.2, @EqWith.WithSelf.NotEqual(%U.loc34_12.1, %T.loc34_21.1) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc34.%T.as_type.loc34_42.1 (%T.as_type), %y.param: @EqWith.loc34.%U.loc34_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc36: @EqWith.loc34.%T.as_type.loc34_42.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc36: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc36: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc36_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.f86, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref Equal, %.loc36_5.1 [symbolic = %assoc0 (constants.%assoc0.5a05b6.2)]
// CHECK:STDOUT:     %impl.elem0.loc36_5.1: @EqWith.loc34.%.loc36_5.2 (%.161b4c.2) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0.52b4ea.2)]
// CHECK:STDOUT:     %bound_method.loc36_5.1: <bound method> = bound_method %x.ref.loc36, %impl.elem0.loc36_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc36_5.1: <specific function> = specific_impl_function %impl.elem0.loc36_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.2d3bf0.2)]
// CHECK:STDOUT:     %bound_method.loc36_5.2: <bound method> = bound_method %x.ref.loc36, %specific_impl_fn.loc36_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc36_5.2(%x.ref.loc36, %y.ref.loc36)
// CHECK:STDOUT:     %x.ref.loc37: @EqWith.loc34.%T.as_type.loc34_42.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc37: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc37: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_36.1 (constants.%EqWith.type.686248.2)]
// CHECK:STDOUT:     %.loc37_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = specific_constant imports.%Core.import_ref.91b, @EqWith.WithSelf(constants.%U, constants.%Self.bd18d5.1) [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.92e927.2) = name_ref NotEqual, %.loc37_5.1 [symbolic = %assoc1 (constants.%assoc1.09a898.2)]
// CHECK:STDOUT:     %impl.elem1.loc37_5.1: @EqWith.loc34.%.loc37_5.2 (%.814) = impl_witness_access constants.%EqWith.lookup_impl_witness.ae2c3a.2, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc37_5.1: <bound method> = bound_method %x.ref.loc37, %impl.elem1.loc37_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc37_5.1: <specific function> = specific_impl_function %impl.elem1.loc37_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.0b5)]
// CHECK:STDOUT:     %bound_method.loc37_5.2: <bound method> = bound_method %x.ref.loc37, %specific_impl_fn.loc37_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc37_5.2(%x.ref.loc37, %y.ref.loc37)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.686248.2
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.295
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.738
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.d13aea.2
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.1bf
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.c79
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B1, constants.%EqWith.facet.c0f) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B1
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.10c
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.e52
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.d5d
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%EqWith.facet.c0f
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.c7f
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.efd
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.9d7
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.9fc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.1e8
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.d5f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e9c
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.b39
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.a54
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.908
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.39e
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.7ce
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.c68
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A1, constants.%EqWith.facet.56f) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A1
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.49a
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.86e
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.4c6
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%EqWith.facet.56f
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.bff
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.033
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.257
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.ab9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.2d1
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.e22
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ac3
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.872
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.cb1
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.abd
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8af
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.bdb
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.15a
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A2, constants.%EqWith.facet.1c6) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A2
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.6a6
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.8fb
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.0c0
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%EqWith.facet.1c6
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.d1e
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.0e6
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.d0b
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.9f0
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.a4f
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.a96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.5d1
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.491
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.341
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.681
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.97f
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.b31
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0c5
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.616
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.6e3
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B3, constants.%EqWith.facet.c7d) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B3
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.131
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.b8d
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.135
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%EqWith.facet.c7d
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.429
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.39e
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.b51
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.e2b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.7da
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.a72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.756
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.a47
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.efa
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.910
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0c8
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.577
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.208
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A3, constants.%EqWith.facet.43a) {
// CHECK:STDOUT:   %U.patt.loc34_12.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A3
// CHECK:STDOUT:   %EqWith.type.loc34_36.1 => constants.%EqWith.type.386
// CHECK:STDOUT:   %pattern_type.loc34_21 => constants.%pattern_type.9d5
// CHECK:STDOUT:   %T.patt.loc34_21.2 => constants.%T.patt.352
// CHECK:STDOUT:   %T.loc34_21.1 => constants.%EqWith.facet.43a
// CHECK:STDOUT:   %T.as_type.loc34_42.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc34_40 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %x.param_patt.loc34_40.2 => constants.%x.param_patt.074
// CHECK:STDOUT:   %x.patt.loc34_40.2 => constants.%x.patt.9ae
// CHECK:STDOUT:   %pattern_type.loc34_46 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %y.param_patt.loc34_46.2 => constants.%y.param_patt.d86
// CHECK:STDOUT:   %y.patt.loc34_46.2 => constants.%y.patt.426
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_40 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_46 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.05b
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.532
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e81
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.4e8
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.65c
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.7d4
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a50
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.0c8
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.82c
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
