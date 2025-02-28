// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/snegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/snegate.carbon

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.snegate";

var arr: array(i32, Negate(Negate(123)));
let arr_p: array(i32, 123)* = &arr;

let n: i32 = Negate(1);

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Negate(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.snegate";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Negate(0)) as Expect(0);
  Test(Negate(1)) as Expect(0 - 1);
  Test(Negate(0 - 0x8000_0000_0000_0000)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.snegate";
fn JustRight(a: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(i32, TooFew());
// CHECK:STDERR:                         ^~~~~~~~
// CHECK:STDERR:
var too_few: array(i32, TooFew());
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(i32, TooMany(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(i32, TooMany(1, 2));
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(i32, BadReturnType(1));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(i32, BadReturnType(1));

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:26: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(i32, JustRight(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(i32, JustRight(1, 2));

fn RuntimeCallIsValidTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallIsValidTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallIsValidBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Negate(a: i32) -> i32 = "int.snegate";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFF_FFFF));

// -INT_MIN is too large for i32.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in negation of -2147483648 [CompileTimeIntegerNegateOverflow]
// CHECK:STDERR: let b: i32 = Negate(-0x8000_0000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Negate(-0x8000_0000);
