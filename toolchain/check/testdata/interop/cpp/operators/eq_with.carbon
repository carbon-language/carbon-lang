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

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
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

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:20:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn NotEqual[self: Self](other: Other) -> bool;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-18]]:8: error: cannot implicitly convert expression of type `Core.Optional(i32* as Core.OptionalStorage)` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator==(EqBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-21]]:8: note: type `Core.Optional(i32* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator==(EqBoolish) const -> int*;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:19:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Equal[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+30]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-22]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator==(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-25]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator==(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:19:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn Equal[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+17]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-34]]:8: error: cannot implicitly convert expression of type `Cpp.Result` to `bool` [ConversionFailure]
// CHECK:STDERR:   auto operator!=(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE-37]]:8: note: type `Cpp.Result` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:   auto operator!=(BothBoolish) const -> Result;
// CHECK:STDERR:        ^
// CHECK:STDERR: {{.*}}/prelude/operators/comparison.carbon:20:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn NotEqual[self: Self](other: Other) -> bool;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_returns_boolish.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
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

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(default_eq_only, default_eq_only);

  EqWith(default_spaceship_only, default_spaceship_only);

  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(only_eq_declared, only_eq_declared);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_comparison_rewrites.carbon:[[@LINE-26]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
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
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(no_relevant_members, no_relevant_members);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EqReturnsVoid` into type implementing `Core.EqWith(Cpp.EqReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(eq_returns_void, eq_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(eq_returns_void, eq_returns_void);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NeqReturnsVoid` into type implementing `Core.EqWith(Cpp.NeqReturnsVoid)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(neq_returns_void, neq_returns_void);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(deleted_neq, deleted_neq);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `Rewriting operator!= using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   EqWith(non_const_eq, non_const_eq);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-76]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(non_const_eq, non_const_eq);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.A2` into type implementing `Core.EqWith(Cpp.B2)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EqWith(a2, b2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-85]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EqWith(a2, b2);
}

// CHECK:STDOUT: --- returns_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d94ae0.1: %EqWith.type.98ea86.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.b4230d.1: %EqWith.WithSelf.NotEqual.type.c5b45a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.ae3c3e.1: %EqWith.WithSelf.Equal.type.b82ec7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.3d4: type = pattern_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.98ea86.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.eee873.2: %EqWith.assoc_type.85b122.2 = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic]
// CHECK:STDOUT:   %assoc1.0ffc88.2: %EqWith.assoc_type.85b122.2 = assoc_entity element1, imports.%Core.import_ref.03f [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.e6374a.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fb5: <witness> = require_complete_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d00: %EqWith.assoc_type.85b122.1 = assoc_entity element0, imports.%Core.import_ref.2e5 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.74c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b82ec7.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.74c = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.67a: <specific function> = specific_impl_function %impl.elem0, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.d80: %EqWith.assoc_type.85b122.1 = assoc_entity element1, imports.%Core.import_ref.b78 [symbolic]
// CHECK:STDOUT:   %.8ed: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c5b45a.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.8ed = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.81f: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqualityComparable: type = class_type @EqualityComparable [concrete]
// CHECK:STDOUT:   %pattern_type.73b: type = pattern_type %EqualityComparable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EqWith.type.6a7: type = facet_type <@EqWith.1, @EqWith.1(%EqualityComparable)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.ecd: type = assoc_entity_type @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %assoc0.fab: %EqWith.assoc_type.ecd = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.c5e: %EqWith.assoc_type.ecd = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal.type: type = fn_type @EqualityComparable.Equal [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal: %EqualityComparable.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual.type: type = fn_type @EqualityComparable.NotEqual [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual: %EqualityComparable.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.505: <witness> = custom_witness (%EqualityComparable.Equal, %EqualityComparable.NotEqual), @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %EqWith.facet.b15: %EqWith.type.6a7 = facet_value %EqualityComparable, (%custom_witness.505) [concrete]
// CHECK:STDOUT:   %pattern_type.876: type = pattern_type %EqWith.type.6a7 [concrete]
// CHECK:STDOUT:   %complete_type.c86: <witness> = complete_type_witness %EqWith.type.6a7 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.6ae: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.b15) [concrete]
// CHECK:STDOUT:   %.96d: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.6ae, %EqWith.facet.b15 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.92a: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.b15) [concrete]
// CHECK:STDOUT:   %.a35: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.92a, %EqWith.facet.b15 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.725: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.d00)]
// CHECK:STDOUT:   %Core.import_ref.8d7: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.d80)]
// CHECK:STDOUT:   %Core.import_ref.03f: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.c5b45a.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.b4230d.1)]
// CHECK:STDOUT:   %Core.import_ref.5ba: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.b82ec7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.ae3c3e.1)]
// CHECK:STDOUT:   %Core.import_ref.2e5 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b78 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc11(%U.loc11_12.2: type, %T.loc11_22.2: @EqWith.loc11.%EqWith.type.loc11_38.1 (%EqWith.type.98ea86.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %EqWith.type.loc11_38.1 [symbolic = %require_complete.loc13 (constants.%require_complete.fb5)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.85b122.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.b82ec7.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc11_22.1 [symbolic = %.loc13_5.2 (constants.%.74c)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_22.1, @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @EqWith.loc11.%.loc13_5.2 (%.74c) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @EqWith.WithSelf.Equal(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element1, imports.%Core.import_ref.03f [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.c5b45a.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc11_22.1 [symbolic = %.loc14_5.2 (constants.%.8ed)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @EqWith.loc11.%.loc14_5.2 (%.8ed) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @EqWith.WithSelf.NotEqual(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type), %y.param: @EqWith.loc11.%U.loc11_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc13: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc13_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.725, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref Equal, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @EqWith.loc11.%.loc13_5.2 (%.74c) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc14: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc14_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.8d7, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref NotEqual, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @EqWith.loc11.%.loc14_5.2 (%.8ed) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc11_38.1 => constants.%EqWith.type.98ea86.2
// CHECK:STDOUT:   %T.loc11_22.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc11_22 => constants.%pattern_type.3d4
// CHECK:STDOUT:   %T.as_type.loc11_44.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc11_42 => constants.%pattern_type.e6374a.2
// CHECK:STDOUT:   %pattern_type.loc11_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%EqualityComparable, constants.%EqWith.facet.b15) {
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %EqWith.type.loc11_38.1 => constants.%EqWith.type.6a7
// CHECK:STDOUT:   %T.loc11_22.1 => constants.%EqWith.facet.b15
// CHECK:STDOUT:   %pattern_type.loc11_22 => constants.%pattern_type.876
// CHECK:STDOUT:   %T.as_type.loc11_44.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %pattern_type.loc11_42 => constants.%pattern_type.73b
// CHECK:STDOUT:   %pattern_type.loc11_48 => constants.%pattern_type.73b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13 => constants.%complete_type.c86
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.ecd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fab
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.6ae
// CHECK:STDOUT:   %.loc13_5.2 => constants.%.96d
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.505
// CHECK:STDOUT:   %impl.elem0.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c5e
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.92a
// CHECK:STDOUT:   %.loc14_5.2 => constants.%.a35
// CHECK:STDOUT:   %impl.elem1.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_equality.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d94ae0.1: %EqWith.type.98ea86.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.b4230d.1: %EqWith.WithSelf.NotEqual.type.c5b45a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.d94ae0.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.ae3c3e.1: %EqWith.WithSelf.Equal.type.b82ec7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.98ea86.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.3d4: type = pattern_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.98ea86.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.85b122.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.eee873.2: %EqWith.assoc_type.85b122.2 = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic]
// CHECK:STDOUT:   %assoc1.0ffc88.2: %EqWith.assoc_type.85b122.2 = assoc_entity element1, imports.%Core.import_ref.03f [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.e6374a.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.fb5: <witness> = require_complete_type %EqWith.type.98ea86.2 [symbolic]
// CHECK:STDOUT:   %assoc0.d00: %EqWith.assoc_type.85b122.1 = assoc_entity element0, imports.%Core.import_ref.2e5 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b82ec7.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.c5b45a.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.74c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b82ec7.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.74c = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.67a: <specific function> = specific_impl_function %impl.elem0, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.d80: %EqWith.assoc_type.85b122.1 = assoc_entity element1, imports.%Core.import_ref.b78 [symbolic]
// CHECK:STDOUT:   %.8ed: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.c5b45a.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.8ed = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.81f: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
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
// CHECK:STDOUT:   %EqWith.type.6ed: type = facet_type <@EqWith.1, @EqWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.b38: type = assoc_entity_type @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.a48: %EqWith.assoc_type.b38 = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.2d4: %EqWith.assoc_type.b38 = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %A1.Equal.type: type = fn_type @A1.Equal [concrete]
// CHECK:STDOUT:   %A1.Equal: %A1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.NotEqual.type: type = fn_type @A1.NotEqual [concrete]
// CHECK:STDOUT:   %A1.NotEqual: %A1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.087: <witness> = custom_witness (%A1.Equal, %A1.NotEqual), @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.2d3: %EqWith.type.6ed = facet_value %A1, (%custom_witness.087) [concrete]
// CHECK:STDOUT:   %pattern_type.e16: type = pattern_type %EqWith.type.6ed [concrete]
// CHECK:STDOUT:   %EqWith.type.dda: type = facet_type <@EqWith.1, @EqWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.5ce: type = assoc_entity_type @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.7c8: %EqWith.assoc_type.5ce = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.651: %EqWith.assoc_type.5ce = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %B1.Equal.type: type = fn_type @B1.Equal [concrete]
// CHECK:STDOUT:   %B1.Equal: %B1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.NotEqual.type: type = fn_type @B1.NotEqual [concrete]
// CHECK:STDOUT:   %B1.NotEqual: %B1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.123: <witness> = custom_witness (%B1.Equal, %B1.NotEqual), @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.b94: %EqWith.type.dda = facet_value %B1, (%custom_witness.123) [concrete]
// CHECK:STDOUT:   %pattern_type.a95: type = pattern_type %EqWith.type.dda [concrete]
// CHECK:STDOUT:   %EqWith.type.c63: type = facet_type <@EqWith.1, @EqWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.2ca: type = assoc_entity_type @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.878: %EqWith.assoc_type.2ca = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.7a2: %EqWith.assoc_type.2ca = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %B2.Equal.type: type = fn_type @B2.Equal [concrete]
// CHECK:STDOUT:   %B2.Equal: %B2.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.NotEqual.type: type = fn_type @B2.NotEqual [concrete]
// CHECK:STDOUT:   %B2.NotEqual: %B2.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.640: <witness> = custom_witness (%B2.Equal, %B2.NotEqual), @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %EqWith.facet.fe0: %EqWith.type.c63 = facet_value %B2, (%custom_witness.640) [concrete]
// CHECK:STDOUT:   %pattern_type.cbd: type = pattern_type %EqWith.type.c63 [concrete]
// CHECK:STDOUT:   %EqWith.type.2e6: type = facet_type <@EqWith.1, @EqWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.eda: type = assoc_entity_type @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.34f: %EqWith.assoc_type.eda = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.c2b: %EqWith.assoc_type.eda = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %A3.Equal.type: type = fn_type @A3.Equal [concrete]
// CHECK:STDOUT:   %A3.Equal: %A3.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.528: <witness> = custom_witness (%A3.Equal, %NotEqual), @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.34d: %EqWith.type.2e6 = facet_value %A3, (%custom_witness.528) [concrete]
// CHECK:STDOUT:   %pattern_type.2b2: type = pattern_type %EqWith.type.2e6 [concrete]
// CHECK:STDOUT:   %EqWith.type.831: type = facet_type <@EqWith.1, @EqWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.d85: type = assoc_entity_type @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.b7b: %EqWith.assoc_type.d85 = assoc_entity element0, imports.%Core.import_ref.5ba [concrete]
// CHECK:STDOUT:   %assoc1.a3b: %EqWith.assoc_type.d85 = assoc_entity element1, imports.%Core.import_ref.03f [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.NotEqual.type: type = fn_type @B3.NotEqual [concrete]
// CHECK:STDOUT:   %B3.NotEqual: %B3.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0ff: <witness> = custom_witness (%Equal, %B3.NotEqual), @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.2a4: %EqWith.type.831 = facet_value %B3, (%custom_witness.0ff) [concrete]
// CHECK:STDOUT:   %pattern_type.9c8: type = pattern_type %EqWith.type.831 [concrete]
// CHECK:STDOUT:   %complete_type.1d5: <witness> = complete_type_witness %EqWith.type.6ed [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.b99: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B1, %EqWith.facet.2d3) [concrete]
// CHECK:STDOUT:   %.bad: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.b99, %EqWith.facet.2d3 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.b2d: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B1, %EqWith.facet.2d3) [concrete]
// CHECK:STDOUT:   %.1c3: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.b2d, %EqWith.facet.2d3 [concrete]
// CHECK:STDOUT:   %complete_type.f84: <witness> = complete_type_witness %EqWith.type.dda [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.97e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A1, %EqWith.facet.b94) [concrete]
// CHECK:STDOUT:   %.f78: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.97e, %EqWith.facet.b94 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a75: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A1, %EqWith.facet.b94) [concrete]
// CHECK:STDOUT:   %.eec: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a75, %EqWith.facet.b94 [concrete]
// CHECK:STDOUT:   %complete_type.f0c: <witness> = complete_type_witness %EqWith.type.c63 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.e2e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A2, %EqWith.facet.fe0) [concrete]
// CHECK:STDOUT:   %.ffe: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.e2e, %EqWith.facet.fe0 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.902: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A2, %EqWith.facet.fe0) [concrete]
// CHECK:STDOUT:   %.609: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.902, %EqWith.facet.fe0 [concrete]
// CHECK:STDOUT:   %complete_type.777: <witness> = complete_type_witness %EqWith.type.2e6 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.a57: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B3, %EqWith.facet.34d) [concrete]
// CHECK:STDOUT:   %.fc2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.a57, %EqWith.facet.34d [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.9c3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B3, %EqWith.facet.34d) [concrete]
// CHECK:STDOUT:   %.b37: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.9c3, %EqWith.facet.34d [concrete]
// CHECK:STDOUT:   %complete_type.94e: <witness> = complete_type_witness %EqWith.type.831 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.351: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A3, %EqWith.facet.2a4) [concrete]
// CHECK:STDOUT:   %.06c: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.351, %EqWith.facet.2a4 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.682: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A3, %EqWith.facet.2a4) [concrete]
// CHECK:STDOUT:   %.3ea: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.682, %EqWith.facet.2a4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.725: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.d00)]
// CHECK:STDOUT:   %Core.import_ref.8d7: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.85b122.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.d80)]
// CHECK:STDOUT:   %Core.import_ref.03f: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.c5b45a.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.b4230d.1)]
// CHECK:STDOUT:   %Core.import_ref.5ba: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.b82ec7.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.ae3c3e.1)]
// CHECK:STDOUT:   %Core.import_ref.2e5 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b78 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc34(%U.loc34_12.2: type, %T.loc34_22.2: @EqWith.loc34.%EqWith.type.loc34_38.1 (%EqWith.type.98ea86.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc36: <witness> = require_complete_type %EqWith.type.loc34_38.1 [symbolic = %require_complete.loc36 (constants.%require_complete.fb5)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.85b122.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element0, imports.%Core.import_ref.5ba [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.b82ec7.3)]
// CHECK:STDOUT:   %.loc36_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc34_22.1 [symbolic = %.loc36_5.2 (constants.%.74c)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc34_22.1, @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc36_5.2: @EqWith.loc34.%.loc36_5.2 (%.74c) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2: <specific function> = specific_impl_function %impl.elem0.loc36_5.2, @EqWith.WithSelf.Equal(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = assoc_entity element1, imports.%Core.import_ref.03f [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.c5b45a.3)]
// CHECK:STDOUT:   %.loc37_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc34_22.1 [symbolic = %.loc37_5.2 (constants.%.8ed)]
// CHECK:STDOUT:   %impl.elem1.loc37_5.2: @EqWith.loc34.%.loc37_5.2 (%.8ed) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2: <specific function> = specific_impl_function %impl.elem1.loc37_5.2, @EqWith.WithSelf.NotEqual(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type), %y.param: @EqWith.loc34.%U.loc34_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc36: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc36: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc36: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc36_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.725, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref Equal, %.loc36_5.1 [symbolic = %assoc0 (constants.%assoc0.eee873.2)]
// CHECK:STDOUT:     %impl.elem0.loc36_5.1: @EqWith.loc34.%.loc36_5.2 (%.74c) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc36_5.1: <bound method> = bound_method %x.ref.loc36, %impl.elem0.loc36_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc36_5.1: <specific function> = specific_impl_function %impl.elem0.loc36_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.67a)]
// CHECK:STDOUT:     %bound_method.loc36_5.2: <bound method> = bound_method %x.ref.loc36, %specific_impl_fn.loc36_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc36_5.2(%x.ref.loc36, %y.ref.loc36)
// CHECK:STDOUT:     %x.ref.loc37: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc37: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc37: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_38.1 (constants.%EqWith.type.98ea86.2)]
// CHECK:STDOUT:     %.loc37_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = specific_constant imports.%Core.import_ref.8d7, @EqWith.WithSelf(constants.%U, constants.%Self.d94ae0.1) [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.85b122.2) = name_ref NotEqual, %.loc37_5.1 [symbolic = %assoc1 (constants.%assoc1.0ffc88.2)]
// CHECK:STDOUT:     %impl.elem1.loc37_5.1: @EqWith.loc34.%.loc37_5.2 (%.8ed) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc37_5.1: <bound method> = bound_method %x.ref.loc37, %impl.elem1.loc37_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc37_5.1: <specific function> = specific_impl_function %impl.elem1.loc37_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.81f)]
// CHECK:STDOUT:     %bound_method.loc37_5.2: <bound method> = bound_method %x.ref.loc37, %specific_impl_fn.loc37_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc37_5.2(%x.ref.loc37, %y.ref.loc37)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.98ea86.2
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.3d4
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.e6374a.2
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B1, constants.%EqWith.facet.2d3) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B1
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.6ed
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.2d3
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.e16
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.3b4
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.5b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.1d5
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.b38
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a48
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.b99
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.bad
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.087
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2d4
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.b2d
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.1c3
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A1, constants.%EqWith.facet.b94) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A1
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.dda
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.b94
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.a95
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.5b2
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.3b4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.f84
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.5ce
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c8
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.97e
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.f78
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.123
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.651
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.a75
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.eec
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A2, constants.%EqWith.facet.fe0) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A2
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.c63
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.fe0
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.cbd
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.d1e
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.9f0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.f0c
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.2ca
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.878
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.e2e
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.ffe
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.640
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7a2
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.902
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.609
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B3, constants.%EqWith.facet.34d) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B3
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.2e6
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.34d
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.2b2
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.7fe
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.2dd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.777
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.eda
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.34f
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.a57
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.fc2
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.528
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c2b
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.9c3
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.b37
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A3, constants.%EqWith.facet.2a4) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A3
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.831
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.2a4
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.9c8
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.7fe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.94e
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.d85
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b7b
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.351
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.06c
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.0ff
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a3b
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.682
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.3ea
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
