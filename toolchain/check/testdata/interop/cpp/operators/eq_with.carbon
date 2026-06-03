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
// CHECK:STDOUT:   %EqWith.type.e412df.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1e4d80.1: %EqWith.type.e412df.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.34d150.1: %EqWith.WithSelf.NotEqual.type.a34bef.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.6ee1cd.1: %EqWith.WithSelf.Equal.type.28b02d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.e412df.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.243: type = pattern_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.e412df.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.04df8c.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element0, imports.%Core.import_ref.eea [symbolic]
// CHECK:STDOUT:   %assoc1.47b841.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element1, imports.%Core.import_ref.228 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.479039.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.1c9: <witness> = require_complete_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5b6: %EqWith.assoc_type.27e9f5.1 = assoc_entity element0, imports.%Core.import_ref.c77 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.e7e: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.28b02d.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e7e = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ba9: <specific function> = specific_impl_function %impl.elem0, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.491: %EqWith.assoc_type.27e9f5.1 = assoc_entity element1, imports.%Core.import_ref.c36 [symbolic]
// CHECK:STDOUT:   %.a13: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a34bef.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.a13 = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a6f: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqualityComparable: type = class_type @EqualityComparable [concrete]
// CHECK:STDOUT:   %pattern_type.51b: type = pattern_type %EqualityComparable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %EqWith.type.dae: type = facet_type <@EqWith.1, @EqWith.1(%EqualityComparable)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.fa0: type = assoc_entity_type @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %assoc0.5f5: %EqWith.assoc_type.fa0 = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.8e8: %EqWith.assoc_type.fa0 = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal.type: type = fn_type @EqualityComparable.Equal [concrete]
// CHECK:STDOUT:   %EqualityComparable.Equal: %EqualityComparable.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual.type: type = fn_type @EqualityComparable.NotEqual [concrete]
// CHECK:STDOUT:   %EqualityComparable.NotEqual: %EqualityComparable.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c57: <witness> = custom_witness (%EqualityComparable.Equal, %EqualityComparable.NotEqual), @EqWith.1, @EqWith.1(%EqualityComparable) [concrete]
// CHECK:STDOUT:   %EqWith.facet.081: %EqWith.type.dae = facet_value %EqualityComparable, (%custom_witness.c57) [concrete]
// CHECK:STDOUT:   %pattern_type.495: type = pattern_type %EqWith.type.dae [concrete]
// CHECK:STDOUT:   %complete_type.c58: <witness> = complete_type_witness %EqWith.type.dae [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.c97: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.081) [concrete]
// CHECK:STDOUT:   %.a66: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.c97, %EqWith.facet.081 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.ecf: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%EqualityComparable, %EqWith.facet.081) [concrete]
// CHECK:STDOUT:   %.87a: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.ecf, %EqWith.facet.081 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0a8: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.5b6)]
// CHECK:STDOUT:   %Core.import_ref.33b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.491)]
// CHECK:STDOUT:   %Core.import_ref.228: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.a34bef.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.34d150.1)]
// CHECK:STDOUT:   %Core.import_ref.eea: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.28b02d.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.6ee1cd.1)]
// CHECK:STDOUT:   %Core.import_ref.c77 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c36 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc11(%U.loc11_12.2: type, %T.loc11_22.2: @EqWith.loc11.%EqWith.type.loc11_38.1 (%EqWith.type.e412df.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %EqWith.type.loc11_38.1 [symbolic = %require_complete.loc13 (constants.%require_complete.1c9)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.27e9f5.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element0, imports.%Core.import_ref.eea [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.28b02d.3)]
// CHECK:STDOUT:   %.loc13_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc11_22.1 [symbolic = %.loc13_5.2 (constants.%.e7e)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc11_22.1, @EqWith.1, @EqWith.1(%U.loc11_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc13_5.2: @EqWith.loc11.%.loc13_5.2 (%.e7e) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2: <specific function> = specific_impl_function %impl.elem0.loc13_5.2, @EqWith.WithSelf.Equal(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element1, imports.%Core.import_ref.228 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.a34bef.3)]
// CHECK:STDOUT:   %.loc14_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc11_22.1 [symbolic = %.loc14_5.2 (constants.%.a13)]
// CHECK:STDOUT:   %impl.elem1.loc14_5.2: @EqWith.loc11.%.loc14_5.2 (%.a13) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2: <specific function> = specific_impl_function %impl.elem1.loc14_5.2, @EqWith.WithSelf.NotEqual(%U.loc11_12.1, %T.loc11_22.1) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type), %y.param: @EqWith.loc11.%U.loc11_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc13: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc13: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc13: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc13_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.0a8, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref Equal, %.loc13_5.1 [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %impl.elem0.loc13_5.1: @EqWith.loc11.%.loc13_5.2 (%.e7e) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc13_5.1: <bound method> = bound_method %x.ref.loc13, %impl.elem0.loc13_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_5.1: <specific function> = specific_impl_function %impl.elem0.loc13_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc13_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:     %bound_method.loc13_5.2: <bound method> = bound_method %x.ref.loc13, %specific_impl_fn.loc13_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc13_5.2(%x.ref.loc13, %y.ref.loc13)
// CHECK:STDOUT:     %x.ref.loc14: @EqWith.loc11.%T.as_type.loc11_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc14: @EqWith.loc11.%U.loc11_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc14: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc11_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc14_5.1: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.33b, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc11.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref NotEqual, %.loc14_5.1 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %impl.elem1.loc14_5.1: @EqWith.loc11.%.loc14_5.2 (%.a13) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc14_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc14_5.1: <bound method> = bound_method %x.ref.loc14, %impl.elem1.loc14_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc14_5.1: <specific function> = specific_impl_function %impl.elem1.loc14_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc14_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:     %bound_method.loc14_5.2: <bound method> = bound_method %x.ref.loc14, %specific_impl_fn.loc14_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc14_5.2(%x.ref.loc14, %y.ref.loc14)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc11_38.1 => constants.%EqWith.type.e412df.2
// CHECK:STDOUT:   %T.loc11_22.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc11_22 => constants.%pattern_type.243
// CHECK:STDOUT:   %T.as_type.loc11_44.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc11_42 => constants.%pattern_type.479039.2
// CHECK:STDOUT:   %pattern_type.loc11_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc11(constants.%EqualityComparable, constants.%EqWith.facet.081) {
// CHECK:STDOUT:   %U.loc11_12.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %EqWith.type.loc11_38.1 => constants.%EqWith.type.dae
// CHECK:STDOUT:   %T.loc11_22.1 => constants.%EqWith.facet.081
// CHECK:STDOUT:   %pattern_type.loc11_22 => constants.%pattern_type.495
// CHECK:STDOUT:   %T.as_type.loc11_44.1 => constants.%EqualityComparable
// CHECK:STDOUT:   %pattern_type.loc11_42 => constants.%pattern_type.51b
// CHECK:STDOUT:   %pattern_type.loc11_48 => constants.%pattern_type.51b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13 => constants.%complete_type.c58
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.fa0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5f5
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.c97
// CHECK:STDOUT:   %.loc13_5.2 => constants.%.a66
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.c57
// CHECK:STDOUT:   %impl.elem0.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc13_5.2 => constants.%EqualityComparable.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8e8
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.ecf
// CHECK:STDOUT:   %.loc14_5.2 => constants.%.87a
// CHECK:STDOUT:   %impl.elem1.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc14_5.2 => constants.%EqualityComparable.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cross_type_equality.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %EqWith.type.e412df.1: type = facet_type <@EqWith.1, @EqWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1e4d80.1: %EqWith.type.e412df.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.1: type = assoc_entity_type @EqWith.1, @EqWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.1: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.34d150.1: %EqWith.WithSelf.NotEqual.type.a34bef.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.1: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%Other, %Self.1e4d80.1) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.6ee1cd.1: %EqWith.WithSelf.Equal.type.28b02d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %EqWith.type.e412df.2: type = facet_type <@EqWith.1, @EqWith.1(%U)> [symbolic]
// CHECK:STDOUT:   %pattern_type.243: type = pattern_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %T: %EqWith.type.e412df.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %EqWith.assoc_type.27e9f5.2: type = assoc_entity_type @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.04df8c.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element0, imports.%Core.import_ref.eea [symbolic]
// CHECK:STDOUT:   %assoc1.47b841.2: %EqWith.assoc_type.27e9f5.2 = assoc_entity element1, imports.%Core.import_ref.228 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.479039.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.1c9: <witness> = require_complete_type %EqWith.type.e412df.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5b6: %EqWith.assoc_type.27e9f5.1 = assoc_entity element0, imports.%Core.import_ref.c77 [symbolic]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T, @EqWith.1, @EqWith.1(%U) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.28b02d.3: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.a34bef.3: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U, %T) [symbolic]
// CHECK:STDOUT:   %.e7e: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.28b02d.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.e7e = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ba9: <specific function> = specific_impl_function %impl.elem0, @EqWith.WithSelf.Equal(%U, %T) [symbolic]
// CHECK:STDOUT:   %assoc1.491: %EqWith.assoc_type.27e9f5.1 = assoc_entity element1, imports.%Core.import_ref.c36 [symbolic]
// CHECK:STDOUT:   %.a13: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.a34bef.3, %T [symbolic]
// CHECK:STDOUT:   %impl.elem1: %.a13 = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a6f: <specific function> = specific_impl_function %impl.elem1, @EqWith.WithSelf.NotEqual(%U, %T) [symbolic]
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
// CHECK:STDOUT:   %EqWith.type.073: type = facet_type <@EqWith.1, @EqWith.1(%B1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.1ba: type = assoc_entity_type @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %assoc0.f24: %EqWith.assoc_type.1ba = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.449: %EqWith.assoc_type.1ba = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %A1.Equal.type: type = fn_type @A1.Equal [concrete]
// CHECK:STDOUT:   %A1.Equal: %A1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %A1.NotEqual.type: type = fn_type @A1.NotEqual [concrete]
// CHECK:STDOUT:   %A1.NotEqual: %A1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8a4: <witness> = custom_witness (%A1.Equal, %A1.NotEqual), @EqWith.1, @EqWith.1(%B1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.716: %EqWith.type.073 = facet_value %A1, (%custom_witness.8a4) [concrete]
// CHECK:STDOUT:   %pattern_type.3f8: type = pattern_type %EqWith.type.073 [concrete]
// CHECK:STDOUT:   %EqWith.type.48c: type = facet_type <@EqWith.1, @EqWith.1(%A1)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.22c: type = assoc_entity_type @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %assoc0.265: %EqWith.assoc_type.22c = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.8b1: %EqWith.assoc_type.22c = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %B1.Equal.type: type = fn_type @B1.Equal [concrete]
// CHECK:STDOUT:   %B1.Equal: %B1.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.NotEqual.type: type = fn_type @B1.NotEqual [concrete]
// CHECK:STDOUT:   %B1.NotEqual: %B1.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3b6: <witness> = custom_witness (%B1.Equal, %B1.NotEqual), @EqWith.1, @EqWith.1(%A1) [concrete]
// CHECK:STDOUT:   %EqWith.facet.5f4: %EqWith.type.48c = facet_value %B1, (%custom_witness.3b6) [concrete]
// CHECK:STDOUT:   %pattern_type.fe1: type = pattern_type %EqWith.type.48c [concrete]
// CHECK:STDOUT:   %EqWith.type.d12: type = facet_type <@EqWith.1, @EqWith.1(%A2)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.1d4: type = assoc_entity_type @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %assoc0.00f: %EqWith.assoc_type.1d4 = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.09d: %EqWith.assoc_type.1d4 = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %B2.Equal.type: type = fn_type @B2.Equal [concrete]
// CHECK:STDOUT:   %B2.Equal: %B2.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.NotEqual.type: type = fn_type @B2.NotEqual [concrete]
// CHECK:STDOUT:   %B2.NotEqual: %B2.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.778: <witness> = custom_witness (%B2.Equal, %B2.NotEqual), @EqWith.1, @EqWith.1(%A2) [concrete]
// CHECK:STDOUT:   %EqWith.facet.5f1: %EqWith.type.d12 = facet_value %B2, (%custom_witness.778) [concrete]
// CHECK:STDOUT:   %pattern_type.481: type = pattern_type %EqWith.type.d12 [concrete]
// CHECK:STDOUT:   %EqWith.type.663: type = facet_type <@EqWith.1, @EqWith.1(%B3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.a6b: type = assoc_entity_type @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %assoc0.dc0: %EqWith.assoc_type.a6b = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.877: %EqWith.assoc_type.a6b = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %A3.Equal.type: type = fn_type @A3.Equal [concrete]
// CHECK:STDOUT:   %A3.Equal: %A3.Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.002: <witness> = custom_witness (%A3.Equal, %NotEqual), @EqWith.1, @EqWith.1(%B3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.615: %EqWith.type.663 = facet_value %A3, (%custom_witness.002) [concrete]
// CHECK:STDOUT:   %pattern_type.1cc: type = pattern_type %EqWith.type.663 [concrete]
// CHECK:STDOUT:   %EqWith.type.819: type = facet_type <@EqWith.1, @EqWith.1(%A3)> [concrete]
// CHECK:STDOUT:   %EqWith.assoc_type.735: type = assoc_entity_type @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %assoc0.af9: %EqWith.assoc_type.735 = assoc_entity element0, imports.%Core.import_ref.eea [concrete]
// CHECK:STDOUT:   %assoc1.1f3: %EqWith.assoc_type.735 = assoc_entity element1, imports.%Core.import_ref.228 [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %B3.NotEqual.type: type = fn_type @B3.NotEqual [concrete]
// CHECK:STDOUT:   %B3.NotEqual: %B3.NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.abb: <witness> = custom_witness (%Equal, %B3.NotEqual), @EqWith.1, @EqWith.1(%A3) [concrete]
// CHECK:STDOUT:   %EqWith.facet.59d: %EqWith.type.819 = facet_value %B3, (%custom_witness.abb) [concrete]
// CHECK:STDOUT:   %pattern_type.c1c: type = pattern_type %EqWith.type.819 [concrete]
// CHECK:STDOUT:   %complete_type.ff3: <witness> = complete_type_witness %EqWith.type.073 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.13e: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B1, %EqWith.facet.716) [concrete]
// CHECK:STDOUT:   %.1c2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.13e, %EqWith.facet.716 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.216: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B1, %EqWith.facet.716) [concrete]
// CHECK:STDOUT:   %.0d2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.216, %EqWith.facet.716 [concrete]
// CHECK:STDOUT:   %complete_type.ea8: <witness> = complete_type_witness %EqWith.type.48c [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.d3f: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A1, %EqWith.facet.5f4) [concrete]
// CHECK:STDOUT:   %.1aa: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.d3f, %EqWith.facet.5f4 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.2be: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A1, %EqWith.facet.5f4) [concrete]
// CHECK:STDOUT:   %.891: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.2be, %EqWith.facet.5f4 [concrete]
// CHECK:STDOUT:   %complete_type.75a: <witness> = complete_type_witness %EqWith.type.d12 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.80a: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A2, %EqWith.facet.5f1) [concrete]
// CHECK:STDOUT:   %.cd2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.80a, %EqWith.facet.5f1 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.fca: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A2, %EqWith.facet.5f1) [concrete]
// CHECK:STDOUT:   %.0d7: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.fca, %EqWith.facet.5f1 [concrete]
// CHECK:STDOUT:   %complete_type.4da: <witness> = complete_type_witness %EqWith.type.663 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.e85: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%B3, %EqWith.facet.615) [concrete]
// CHECK:STDOUT:   %.a39: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.e85, %EqWith.facet.615 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.2cb: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%B3, %EqWith.facet.615) [concrete]
// CHECK:STDOUT:   %.fe4: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.2cb, %EqWith.facet.615 [concrete]
// CHECK:STDOUT:   %complete_type.526: <witness> = complete_type_witness %EqWith.type.819 [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type.6b2: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%A3, %EqWith.facet.59d) [concrete]
// CHECK:STDOUT:   %.58b: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type.6b2, %EqWith.facet.59d [concrete]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type.353: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%A3, %EqWith.facet.59d) [concrete]
// CHECK:STDOUT:   %.b5c: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type.353, %EqWith.facet.59d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0a8: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc0 (constants.%assoc0.5b6)]
// CHECK:STDOUT:   %Core.import_ref.33b: @EqWith.WithSelf.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%assoc1 (constants.%assoc1.491)]
// CHECK:STDOUT:   %Core.import_ref.228: @EqWith.WithSelf.%EqWith.WithSelf.NotEqual.type (%EqWith.WithSelf.NotEqual.type.a34bef.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.NotEqual (constants.%EqWith.WithSelf.NotEqual.34d150.1)]
// CHECK:STDOUT:   %Core.import_ref.eea: @EqWith.WithSelf.%EqWith.WithSelf.Equal.type (%EqWith.WithSelf.Equal.type.28b02d.1) = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, loaded [symbolic = @EqWith.WithSelf.%EqWith.WithSelf.Equal (constants.%EqWith.WithSelf.Equal.6ee1cd.1)]
// CHECK:STDOUT:   %Core.import_ref.c77 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c36 = import_ref Core//prelude/operators/comparison, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EqWith.loc34(%U.loc34_12.2: type, %T.loc34_22.2: @EqWith.loc34.%EqWith.type.loc34_38.1 (%EqWith.type.e412df.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc36: <witness> = require_complete_type %EqWith.type.loc34_38.1 [symbolic = %require_complete.loc36 (constants.%require_complete.1c9)]
// CHECK:STDOUT:   %EqWith.assoc_type: type = assoc_entity_type @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.assoc_type (constants.%EqWith.assoc_type.27e9f5.2)]
// CHECK:STDOUT:   %assoc0: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element0, imports.%Core.import_ref.eea [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type: type = fn_type @EqWith.WithSelf.Equal, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %EqWith.WithSelf.Equal.type (constants.%EqWith.WithSelf.Equal.type.28b02d.3)]
// CHECK:STDOUT:   %.loc36_5.2: type = fn_type_with_self_type %EqWith.WithSelf.Equal.type, %T.loc34_22.1 [symbolic = %.loc36_5.2 (constants.%.e7e)]
// CHECK:STDOUT:   %EqWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc34_22.1, @EqWith.1, @EqWith.1(%U.loc34_12.1) [symbolic = %EqWith.lookup_impl_witness (constants.%EqWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc36_5.2: @EqWith.loc34.%.loc36_5.2 (%.e7e) = impl_witness_access %EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2: <specific function> = specific_impl_function %impl.elem0.loc36_5.2, @EqWith.WithSelf.Equal(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:   %assoc1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = assoc_entity element1, imports.%Core.import_ref.228 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type: type = fn_type @EqWith.WithSelf.NotEqual, @EqWith.WithSelf(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %EqWith.WithSelf.NotEqual.type (constants.%EqWith.WithSelf.NotEqual.type.a34bef.3)]
// CHECK:STDOUT:   %.loc37_5.2: type = fn_type_with_self_type %EqWith.WithSelf.NotEqual.type, %T.loc34_22.1 [symbolic = %.loc37_5.2 (constants.%.a13)]
// CHECK:STDOUT:   %impl.elem1.loc37_5.2: @EqWith.loc34.%.loc37_5.2 (%.a13) = impl_witness_access %EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2: <specific function> = specific_impl_function %impl.elem1.loc37_5.2, @EqWith.WithSelf.NotEqual(%U.loc34_12.1, %T.loc34_22.1) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type), %y.param: @EqWith.loc34.%U.loc34_12.1 (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc36: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc36: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc36: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc36_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.0a8, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %Equal.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref Equal, %.loc36_5.1 [symbolic = %assoc0 (constants.%assoc0.04df8c.2)]
// CHECK:STDOUT:     %impl.elem0.loc36_5.1: @EqWith.loc34.%.loc36_5.2 (%.e7e) = impl_witness_access constants.%EqWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc36_5.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc36_5.1: <bound method> = bound_method %x.ref.loc36, %impl.elem0.loc36_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc36_5.1: <specific function> = specific_impl_function %impl.elem0.loc36_5.1, @EqWith.WithSelf.Equal(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc36_5.2 (constants.%specific_impl_fn.ba9)]
// CHECK:STDOUT:     %bound_method.loc36_5.2: <bound method> = bound_method %x.ref.loc36, %specific_impl_fn.loc36_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.Equal.call: init bool = call %bound_method.loc36_5.2(%x.ref.loc36, %y.ref.loc36)
// CHECK:STDOUT:     %x.ref.loc37: @EqWith.loc34.%T.as_type.loc34_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %y.ref.loc37: @EqWith.loc34.%U.loc34_12.1 (%U) = name_ref y, %y
// CHECK:STDOUT:     %EqWith.type.loc37: type = facet_type <@EqWith.1, @EqWith.1(constants.%U)> [symbolic = %EqWith.type.loc34_38.1 (constants.%EqWith.type.e412df.2)]
// CHECK:STDOUT:     %.loc37_5.1: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = specific_constant imports.%Core.import_ref.33b, @EqWith.WithSelf(constants.%U, constants.%Self.1e4d80.1) [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %NotEqual.ref: @EqWith.loc34.%EqWith.assoc_type (%EqWith.assoc_type.27e9f5.2) = name_ref NotEqual, %.loc37_5.1 [symbolic = %assoc1 (constants.%assoc1.47b841.2)]
// CHECK:STDOUT:     %impl.elem1.loc37_5.1: @EqWith.loc34.%.loc37_5.2 (%.a13) = impl_witness_access constants.%EqWith.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc37_5.2 (constants.%impl.elem1)]
// CHECK:STDOUT:     %bound_method.loc37_5.1: <bound method> = bound_method %x.ref.loc37, %impl.elem1.loc37_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc37_5.1: <specific function> = specific_impl_function %impl.elem1.loc37_5.1, @EqWith.WithSelf.NotEqual(constants.%U, constants.%T) [symbolic = %specific_impl_fn.loc37_5.2 (constants.%specific_impl_fn.a6f)]
// CHECK:STDOUT:     %bound_method.loc37_5.2: <bound method> = bound_method %x.ref.loc37, %specific_impl_fn.loc37_5.1
// CHECK:STDOUT:     %EqWith.WithSelf.NotEqual.call: init bool = call %bound_method.loc37_5.2(%x.ref.loc37, %y.ref.loc37)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%U, constants.%T) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%U
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.e412df.2
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%T
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.243
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.479039.2
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B1, constants.%EqWith.facet.716) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B1
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.073
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.716
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.3f8
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%A1
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.c8a
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.e62
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.ff3
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.1ba
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f24
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.13e
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.1c2
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.8a4
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.449
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.216
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.0d2
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%A1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A1, constants.%EqWith.facet.5f4) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A1
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.48c
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.5f4
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.fe1
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B1
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.e62
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.c8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.ea8
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.22c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.265
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.d3f
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.1aa
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.3b6
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B1.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8b1
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.2be
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.891
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B1.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A2, constants.%EqWith.facet.5f1) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A2
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.d12
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.5f1
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.481
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B2
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.fab
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.1e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.75a
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.1d4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.00f
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.80a
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.cd2
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.778
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%B2.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.09d
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.fca
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.0d7
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B2.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%B3, constants.%EqWith.facet.615) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%B3
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.663
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.615
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.1cc
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%A3
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.ada
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.a20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.4da
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.a6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dc0
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.e85
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.a39
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.002
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%A3.Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.877
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.2cb
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.fe4
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EqWith.loc34(constants.%A3, constants.%EqWith.facet.59d) {
// CHECK:STDOUT:   %U.loc34_12.1 => constants.%A3
// CHECK:STDOUT:   %EqWith.type.loc34_38.1 => constants.%EqWith.type.819
// CHECK:STDOUT:   %T.loc34_22.1 => constants.%EqWith.facet.59d
// CHECK:STDOUT:   %pattern_type.loc34_22 => constants.%pattern_type.c1c
// CHECK:STDOUT:   %T.as_type.loc34_44.1 => constants.%B3
// CHECK:STDOUT:   %pattern_type.loc34_42 => constants.%pattern_type.a20
// CHECK:STDOUT:   %pattern_type.loc34_48 => constants.%pattern_type.ada
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34_42 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc34_48 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc36 => constants.%complete_type.526
// CHECK:STDOUT:   %EqWith.assoc_type => constants.%EqWith.assoc_type.735
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.af9
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.type => constants.%EqWith.WithSelf.Equal.type.6b2
// CHECK:STDOUT:   %.loc36_5.2 => constants.%.58b
// CHECK:STDOUT:   %EqWith.lookup_impl_witness => constants.%custom_witness.abb
// CHECK:STDOUT:   %impl.elem0.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %specific_impl_fn.loc36_5.2 => constants.%Equal
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.1f3
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.type => constants.%EqWith.WithSelf.NotEqual.type.353
// CHECK:STDOUT:   %.loc37_5.2 => constants.%.b5c
// CHECK:STDOUT:   %impl.elem1.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT:   %specific_impl_fn.loc37_5.2 => constants.%B3.NotEqual
// CHECK:STDOUT: }
// CHECK:STDOUT:
