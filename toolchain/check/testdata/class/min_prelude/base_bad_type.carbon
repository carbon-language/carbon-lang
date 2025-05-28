// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/base_bad_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/base_bad_type.carbon

// --- fail_derive_from_error.carbon

library "[[@TEST_NAME]]";

class DeriveFromError {
  // CHECK:STDERR: fail_derive_from_error.carbon:[[@LINE+4]]:16: error: name `error` not found [NameNotFound]
  // CHECK:STDERR:   extend base: error;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> {} { return (*p).n; }

// --- fail_derive_from_non_type.carbon

library "[[@TEST_NAME]]";

class DeriveFromNonType {
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+7]]:16: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+4]]:16: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   extend base: 32;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> {} { return (*p).n; }

// --- fail_derive_from_builtin.carbon

library "[[@TEST_NAME]]";

fn Bool() -> type = "bool.make_type";

class DeriveFromBuiltin {
  // CHECK:STDERR: fail_derive_from_builtin.carbon:[[@LINE+4]]:16: error: deriving from final type `bool`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Bool();
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  extend base: Bool();
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_derive_from_builtin.carbon:[[@LINE+7]]:64: error: cannot implicitly convert expression of type `DeriveFromBuiltin*` to `bool*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseBuiltin(p: DeriveFromBuiltin*) -> Bool()* { return p; }
// CHECK:STDERR:                                                                ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_builtin.carbon:[[@LINE+4]]:64: note: type `DeriveFromBuiltin*` does not implement interface `Core.ImplicitAs(bool*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseBuiltin(p: DeriveFromBuiltin*) -> Bool()* { return p; }
// CHECK:STDERR:                                                                ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseBuiltin(p: DeriveFromBuiltin*) -> Bool()* { return p; }

fn AccessMemberWithInvalidBaseBuiltin(p: DeriveFromBuiltin*) -> Bool() { return (*p).n; }

// --- fail_derive_from_tuple.carbon

library "[[@TEST_NAME]]";

base class Base {}

class DeriveFromTuple {
  // CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+4]]:16: error: deriving from final type `(Base,)`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: (Base,);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+7]]:61: error: cannot implicitly convert expression of type `DeriveFromTuple*` to `(Base,)*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+4]]:61: note: type `DeriveFromTuple*` does not implement interface `Core.ImplicitAs((Base,)*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:                                                             ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> {} { return (*p).n; }

// --- fail_derive_from_struct.carbon

library "[[@TEST_NAME]]";

// TODO: Should we allow this?
// We do allow `{.base = {.a: {}, .b: {}}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+4]]:16: error: deriving from final type `{.a: {}, .b: {}}`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: {.a: {}, .b: {}};
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: {.a: {}, .b: {}};
}

// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+7]]:72: error: cannot implicitly convert expression of type `DeriveFromStruct*` to `{.a: {}, .b: {}}*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: {}, .b: {}}* { return p; }
// CHECK:STDERR:                                                                        ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+4]]:72: note: type `DeriveFromStruct*` does not implement interface `Core.ImplicitAs({.a: {}, .b: {}}*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: {}, .b: {}}* { return p; }
// CHECK:STDERR:                                                                        ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: {}, .b: {}}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> {} { return (*p).n; }

// --- fail_derive_from_incomplete.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+4]]:1: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: base class Incomplete;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+7]]:16: error: base `Incomplete` is an incomplete type [IncompleteTypeInBaseDecl]
  // CHECK:STDERR:   extend base: Incomplete;
  // CHECK:STDERR:                ^~~~~~~~~~
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: base class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+7]]:74: error: cannot implicitly convert expression of type `DeriveFromIncomplete*` to `Incomplete*` [ConversionFailure]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+4]]:74: note: type `DeriveFromIncomplete*` does not implement interface `Core.ImplicitAs(Incomplete*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:                                                                          ^~~~~~~~~
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> {} { return (*p).n; }

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

class Final {
  var a: {};
}

class DeriveFromFinal {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:16: error: deriving from final type `Final`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: Final;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  extend base: Final;
}

// For error recovery purposes, we derive from the final type anyway.
fn ConvertToBadBaseFinal(p: DeriveFromFinal*) -> Final* {
  return p;
}

fn AccessMemberWithInvalidBaseFinal_WithMember(p: DeriveFromFinal*) -> {} {
  return (*p).a;
}

fn AccessMemberWithInvalidBaseFinal_NoMember(p: DeriveFromFinal*) -> {} {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+4]]:10: error: member name `b` not found in `DeriveFromFinal` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return (*p).b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return (*p).b;
}
