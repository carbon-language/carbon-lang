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
  // CHECK:STDERR:        ╭─┤ fail_derive_from_error.carbon:[[@LINE+6]]:16
  // CHECK:STDERR: ->     │   extend base: error;
  // CHECK:STDERR:        ·                ━━┳━━
  // CHECK:STDERR:        ·                  ┗━━┫ error: name `error` not found [NameNotFound]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: error;
}

// This should not produce an error.
fn AccessMemberWithInvalidBaseError(p: DeriveFromError*) -> i32 { return (*p).n; }

// --- fail_derive_from_non_type.carbon

library "[[@TEST_NAME]]";

class DeriveFromNonType {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_non_type.carbon:[[@LINE+11]]:16
  // CHECK:STDERR: ->     │   extend base: 32;
  // CHECK:STDERR:        ·                ┳━
  // CHECK:STDERR:        ·                ┗━━┫ error: cannot implicitly convert non-type value of type
  // CHECK:STDERR:        ·                │  ┃        `Core.IntLiteral` to `type`
  // CHECK:STDERR:        ·                │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·                ╰────┤ type `Core.IntLiteral` does not implement interface
  // CHECK:STDERR:        ·                     │ `Core.ImplicitAs(type)`
  // CHECK:STDERR:        ·                     │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: 32;
}

fn AccessMemberWithInvalidBasNonType(p: DeriveFromNonType*) -> i32 { return (*p).n; }

// --- fail_derive_from_i32.carbon

library "[[@TEST_NAME]]";

class DeriveFromi32 {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_i32.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   extend base: i32;
  // CHECK:STDERR:        ·                ┳━━
  // CHECK:STDERR:        ·                ┗━━┫ error: deriving from final type `i32`; base type must be an
  // CHECK:STDERR:        ·                   ┃        `abstract` or `base` class
  // CHECK:STDERR:        ·                   ┃        [BaseIsFinal]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: i32;
}

// It's not really important whether this conversion produces an error or not,
// but it shouldn't crash.
// CHECK:STDERR:        ╭─┤ fail_derive_from_i32.carbon:[[@LINE+16]]:53
// CHECK:STDERR: ->     │ fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }
// CHECK:STDERR:        ·                                                     ┳━━━━━━━━
// CHECK:STDERR:        ·                                                     ┗━━┫ error: cannot
// CHECK:STDERR:        ·                                                     │  ┃        implicitly
// CHECK:STDERR:        ·                                                     │  ┃        convert
// CHECK:STDERR:        ·                                                     │  ┃        expression of
// CHECK:STDERR:        ·                                                     │  ┃        type
// CHECK:STDERR:        ·                                                     │  ┃        `DeriveFromi32*`
// CHECK:STDERR:        ·                                                     │  ┃        to `i32*`
// CHECK:STDERR:        ·                                                     │  ┃        [ConversionFailure]
// CHECK:STDERR:        · ╭───────────────────────────────────────────────────╯
// CHECK:STDERR:        · │ type `DeriveFromi32*` does not implement interface `Core.ImplicitAs(i32*)`
// CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertToBadBasei32(p: DeriveFromi32*) -> i32* { return p; }

// CHECK:STDERR:        ╭─┤ fail_derive_from_i32.carbon:[[@LINE+8]]:70
// CHECK:STDERR: ->     │ fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }
// CHECK:STDERR:        ·                                                                      ━━━┳━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: member name `n` not found in `DeriveFromi32`
// CHECK:STDERR:        · ┃        [MemberNameNotFoundInInstScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn AccessMemberWithInvalidBasei32(p: DeriveFromi32*) -> i32 { return (*p).n; }

// --- fail_derive_from_tuple.carbon

library "[[@TEST_NAME]]";

base class Base {}

class DeriveFromTuple {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_tuple.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   extend base: (Base,);
  // CHECK:STDERR:        ·                ┳━━━━━━
  // CHECK:STDERR:        ·                ┗━━┫ error: deriving from final type `(Base,)`; base type must be
  // CHECK:STDERR:        ·                   ┃        an `abstract` or `base` class
  // CHECK:STDERR:        ·                   ┃        [BaseIsFinal]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: (Base,);
}

// CHECK:STDERR:        ╭─┤ fail_derive_from_tuple.carbon:[[@LINE+13]]:61
// CHECK:STDERR: ->     │ fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }
// CHECK:STDERR:        ·                                                             ━━━━┳━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot implicitly convert expression of type `DeriveFromTuple*` to
// CHECK:STDERR:        · ┃        `(Base,)*`
// CHECK:STDERR:        · ┃        [ConversionFailure]
// CHECK:STDERR:        · ╭───────────────────────────────────────────────────────────────╯
// CHECK:STDERR:        · │ type `DeriveFromTuple*` does not implement interface
// CHECK:STDERR:        · │ `Core.ImplicitAs((Base,)*)`
// CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertToBadBaseTuple(p: DeriveFromTuple*) -> (Base,)* { return p; }

fn AccessMemberWithInvalidBaseTuple(p: DeriveFromTuple*) -> i32 { return (*p).n; }

// --- fail_derive_from_struct.carbon

library "[[@TEST_NAME]]";

// TODO: Should we allow this?
// We do allow `{.base = {.a: i32, .b: i32}}`.
class DeriveFromStruct {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_struct.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   extend base: {.a: i32, .b: i32};
  // CHECK:STDERR:        ·                ┳━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·                ┗━━┫ error: deriving from final type `{.a: i32, .b: i32}`; base
  // CHECK:STDERR:        ·                   ┃        type must be an `abstract` or `base` class
  // CHECK:STDERR:        ·                   ┃        [BaseIsFinal]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: {.a: i32, .b: i32};
}

// CHECK:STDERR:        ╭─┤ fail_derive_from_struct.carbon:[[@LINE+13]]:74
// CHECK:STDERR: ->     │ ...ToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }
// CHECK:STDERR:        ·                                                                   ━━━━┳━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot implicitly convert expression of type `DeriveFromStruct*` to
// CHECK:STDERR:        · ┃        `{.a: i32, .b: i32}*`
// CHECK:STDERR:        · ┃        [ConversionFailure]
// CHECK:STDERR:        · ╭─────────────────────────────────────────────────────────────────────╯
// CHECK:STDERR:        · │ type `DeriveFromStruct*` does not implement interface
// CHECK:STDERR:        · │ `Core.ImplicitAs({.a: i32, .b: i32}*)`
// CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertToBadBaseStruct(p: DeriveFromStruct*) -> {.a: i32, .b: i32}* { return p; }

// It would be OK to reject this if we start actually looking in the struct type.
fn AccessMemberWithInvalidBaseStruct(p: DeriveFromStruct*) -> i32 { return (*p).n; }

// --- fail_derive_from_incomplete.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_derive_from_incomplete.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ base class Incomplete;
// CHECK:STDERR:        · ━━┳━
// CHECK:STDERR:        ·   ┗━━┫ error: `base` not allowed on `class` forward declaration, only definition
// CHECK:STDERR:        ·      ┃        [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
base class Incomplete;

class DeriveFromIncomplete {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_incomplete.carbon:[[@LINE+11]]:16
  // CHECK:STDERR:        │ base class Incomplete;
  // CHECK:STDERR:        · ───────────┬──────────
  // CHECK:STDERR:        ·            ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   extend base: Incomplete;
  // CHECK:STDERR:        ·                ━━━━━┳━━━━
  // CHECK:STDERR:        ·                     ┗━━┫ error: base `Incomplete` is an incomplete type
  // CHECK:STDERR:        ·                        ┃        [IncompleteTypeInBaseDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend base: Incomplete;
}

// CHECK:STDERR:        ╭─┤ fail_derive_from_incomplete.carbon:[[@LINE+13]]:74
// CHECK:STDERR: ->     │ ...ToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }
// CHECK:STDERR:        ·                                                                   ━━━━┳━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: cannot implicitly convert expression of type `DeriveFromIncomplete*` to
// CHECK:STDERR:        · ┃        `Incomplete*`
// CHECK:STDERR:        · ┃        [ConversionFailure]
// CHECK:STDERR:        · ╭─────────────────────────────────────────────────────────────────────╯
// CHECK:STDERR:        · │ type `DeriveFromIncomplete*` does not implement interface
// CHECK:STDERR:        · │ `Core.ImplicitAs(Incomplete*)`
// CHECK:STDERR:        · │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertToBadBaseIncomplete(p: DeriveFromIncomplete*) -> Incomplete* { return p; }

fn AccessMemberWithInvalidBaseIncomplete(p: DeriveFromIncomplete*) -> i32 { return (*p).n; }

// --- fail_derive_from_final.carbon

library "[[@TEST_NAME]]";

class Final {
  var a: i32;
}

class DeriveFromFinal {
  // CHECK:STDERR:        ╭─┤ fail_derive_from_final.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   extend base: Final;
  // CHECK:STDERR:        ·                ┳━━━━
  // CHECK:STDERR:        ·                ┗━━┫ error: deriving from final type `Final`; base type must be
  // CHECK:STDERR:        ·                   ┃        an `abstract` or `base` class
  // CHECK:STDERR:        ·                   ┃        [BaseIsFinal]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_derive_from_final.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   return (*p).b;
  // CHECK:STDERR:        ·          ━━━┳━━
  // CHECK:STDERR:        ·             ┗━━┫ error: member name `b` not found in `DeriveFromFinal`
  // CHECK:STDERR:        ·                ┃        [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return (*p).b;
}
