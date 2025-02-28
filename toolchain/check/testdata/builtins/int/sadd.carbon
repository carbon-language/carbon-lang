// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sadd.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Add(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Add(0, 0)) as Expect(0);
  Test(Add(1, 2)) as Expect(3);

  // Test some cases that might -- but shouldn't -- overflow.
  Test(Add(0x7FFF_FFFE, 1)) as Expect(0x7FFF_FFFF);
  Test(Add(0x7FFF_FFFF, 1)) as Expect(0x8000_0000);
  Test(Add(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0x8000_0000_0000_0000);
  Test(Add(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_0000_0000_0000_0000);
  Test(Add(-0x8000_0000_0000_0000, -1)) as Expect(-0x8000_0000_0000_0001);
  Test(Add(-0x8000_0000_0000_0000, -0x8000_0000_0000_0000)) as Expect(-0x1_0000_0000_0000_0000);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";

// Heterogeneous "add" is not supported.
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAdd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAdd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.sadd";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(i32, TooFew(1));
// CHECK:STDERR:                         ^~~~~~~~~
// CHECK:STDERR:
var too_few: array(i32, TooFew(1));
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(i32, TooMany(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(i32, TooMany(1, 2, 3));
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(i32, BadReturnType(1, 2));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(i32, BadReturnType(1, 2));

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:26: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(i32, JustRight(1, 2, 3));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-43]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(i32, JustRight(1, 2, 3));

fn RuntimeCallIsValidTooFew(a: i32) -> i32 {
  return TooFew(a);
}

fn RuntimeCallIsValidTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

fn RuntimeCallIsValidBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

let a: i32 = Add(0x7FFFFFFF, 0);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `2147483647 + 1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Add(0x7FFFFFFF, 1);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = Add(0x7FFFFFFF, 1);
