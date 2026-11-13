// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/type/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/type/and.carbon

// --- combine_facets.carbon

library "[[@TEST_NAME]]";

fn TypeAnd(a: type, b: type) -> type = "type.and";

interface I {}
interface J {}

fn TakeIJ(unused generic T: TypeAnd(I, J)) {}

class IJ {
  impl as I {}
  impl as J {}
}

fn Call() {
  TakeIJ(IJ);
}

// --- fail_combine_facets_bad_call.carbon

library "[[@TEST_NAME]]";

fn TypeAnd(a: type, b: type) -> type = "type.and";

interface I {}
interface J {}

fn TakeIJ(unused generic T: TypeAnd(I, J)) {}

class JustI {
  impl as I {}
}

class JustJ {
  impl as J {}
}

fn Call() {
  // CHECK:STDERR:        ╭─┤ fail_combine_facets_bad_call.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn TakeIJ(unused generic T: TypeAnd(I, J)) {}
  // CHECK:STDERR:        ·                          ──┬─────────────
  // CHECK:STDERR:        ·                            ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                               │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   TakeIJ(JustI);
  // CHECK:STDERR:        ·   ━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·         ┗━━┫ error: cannot convert type `JustI` into type implementing `I & J`
  // CHECK:STDERR:        ·            ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TakeIJ(JustI);
  // CHECK:STDERR:        ╭─┤ fail_combine_facets_bad_call.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn TakeIJ(unused generic T: TypeAnd(I, J)) {}
  // CHECK:STDERR:        ·                          ──┬─────────────
  // CHECK:STDERR:        ·                            ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                               │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   TakeIJ(JustJ);
  // CHECK:STDERR:        ·   ━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·         ┗━━┫ error: cannot convert type `JustJ` into type implementing `I & J`
  // CHECK:STDERR:        ·            ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  TakeIJ(JustJ);
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_bad_decl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn TooFew(a: type) -> type = "type.and";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━┳━━━━━━━━━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: invalid signature for builtin function "type.and"
// CHECK:STDERR:        ·                  ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn TooFew(a: type) -> type = "type.and";

// CHECK:STDERR:        ╭─┤ fail_bad_decl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn TooMany(a: type, b: type, c: type) -> type = "type.and";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: invalid signature for builtin function "type.and"
// CHECK:STDERR:        ·                       ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn TooMany(a: type, b: type, c: type) -> type = "type.and";

// CHECK:STDERR:        ╭─┤ fail_bad_decl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn ParamNotType(a: {}, b: {}) -> type = "type.and";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: invalid signature for builtin function "type.and"
// CHECK:STDERR:        ·                       ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ParamNotType(a: {}, b: {}) -> type = "type.and";

// CHECK:STDERR:        ╭─┤ fail_bad_decl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn ResultNotType(a: type, b: type) -> {} = "type.and";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: invalid signature for builtin function "type.and"
// CHECK:STDERR:        ·                       ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ResultNotType(a: type, b: type) -> {} = "type.and";

// CHECK:STDERR:        ╭─┤ fail_bad_decl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn ResultNotSpecified(a: type, b: type) = "type.and";
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                    ┗━━┫ error: invalid signature for builtin function "type.and"
// CHECK:STDERR:        ·                       ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ResultNotSpecified(a: type, b: type) = "type.and";
