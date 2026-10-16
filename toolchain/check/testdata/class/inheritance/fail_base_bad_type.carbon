// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_base_bad_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_base_bad_type.carbon

// --- fail_derive_from_error.carbon

library "[[@TEST_NAME]]";

class DeriveFromError {
  // CHECK:STDERR: fail_derive_from_error.carbon:[[@LINE+2]]:16-20: error: name `error` not found [NameNotFound]
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

// --- fail_derive_from_non_type.carbon

library "[[@TEST_NAME]]";

class DeriveFromNonType {
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+3]]:16-17: error: cannot implicitly convert non-type value of type `Core.IntLiteral` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR: fail_derive_from_non_type.carbon:[[@LINE+2]]:16-17: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

// --- fail_derive_from_i32.carbon

library "[[@TEST_NAME]]";

class DeriveFromi32 {
  // CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+2]]:16-18: error: deriving from final type `i32`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+3]]:53-61: error: cannot implicitly convert expression of type `DeriveFromi32*` to `i32*` [ConversionFailure]
// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+2]]:53-61: note: type `DeriveFromi32*` does not implement interface `Core.ImplicitAs(i32*)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

// CHECK:STDERR: fail_derive_from_i32.carbon:[[@LINE+2]]:70-75: error: member name `n` not found in `DeriveFromi32` [MemberNameNotFoundInInstScope]
// CHECK:STDERR:
fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

// --- fail_derive_from_tuple.carbon

library "[[@TEST_NAME]]";

base class Base {}

class DeriveFromTuple {
  // CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+2]]:16-22: error: deriving from final type `(Base,)`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+3]]:61-69: error: cannot implicitly convert expression of type `DeriveFromTuple*` to `(Base,)*` [ConversionFailure]
// CHECK:STDERR: fail_derive_from_tuple.carbon:[[@LINE+2]]:61-69: note: type `DeriveFromTuple*` does not implement interface `Core.ImplicitAs((Base,)*)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// --- fail_derive_from_struct.carbon

library "[[@TEST_NAME]]";

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+2]]:16-33: error: deriving from final type `{.a: i32, .b: i32}`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+3]]:74-82: error: cannot implicitly convert expression of type `DeriveFromStruct*` to `{.a: i32, .b: i32}*` [ConversionFailure]
// CHECK:STDERR: fail_derive_from_struct.carbon:[[@LINE+2]]:74-82: note: type `DeriveFromStruct*` does not implement interface `Core.ImplicitAs({.a: i32, .b: i32}*)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

// --- fail_derive_from_incomplete.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+2]]:1-4: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:
base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+3]]:16-25: error: base `Incomplete` is an incomplete type [IncompleteTypeInBaseDecl]
  // CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE-4]]:1-22: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+3]]:74-82: error: cannot implicitly convert expression of type `DeriveFromIncomplete*` to `Incomplete*` [ConversionFailure]
// CHECK:STDERR: fail_derive_from_incomplete.carbon:[[@LINE+2]]:74-82: note: type `DeriveFromIncomplete*` does not implement interface `Core.ImplicitAs(Incomplete*)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

class Final {
  var a: i32;
}

class DeriveFromFinal {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+2]]:16-20: error: deriving from final type `Final`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:
  extend base: Final;
}

// For error recovery purposes, we derive from the final type anyway.
fn ConvertToBadBaseFinal(p: DeriveFromFinal*) -> Final* {
  return p;
}

fn AccessMemberWithInvalidBaseFinal_WithMember(p: DeriveFromFinal*) -> i32 {
  return (*p).a;
}

fn AccessMemberWithInvalidBaseFinal_NoMember(p: DeriveFromFinal*) -> i32 {
  // CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+2]]:10-15: error: member name `b` not found in `DeriveFromFinal` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  return (*p).b;
}
