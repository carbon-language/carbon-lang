// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/unegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/unegate.carbon

// --- int_negate.carbon

fn Negate(a: u32) -> u32 = "int.unegate";

var arr: array(u32, Negate(Negate(123)));
let arr_p: array(u32, 123)* = &arr;

let n: u32 = Negate(1);

fn RuntimeCallIsValid(a: u32, b: u32) -> u32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> u32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: u32) -> bool = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: u32) -> bool = "int.unegate";
fn JustRight(a: u32) -> u32 = "int.unegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:25: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: array(u32, TooFew());
// CHECK:STDERR:                         ^~~~~~~~
// CHECK:STDERR:
var too_few: array(u32, TooFew());
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:26: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: array(u32, TooMany(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: array(u32, TooMany(1, 2));
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:33: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: array(u32, BadReturnType(1));
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: array(u32, BadReturnType(1));

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:26: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: array(u32, JustRight(1, 2));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: u32) -> u32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: array(u32, JustRight(1, 2));

fn RuntimeCallIsValidTooFew(a: u32) -> u32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> u32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallIsValidTooMany(a: u32, b: u32, c: u32) -> u32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: u32, b: u32) -> u32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallIsValidBadReturnType(a: u32, b: u32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: u32) -> bool = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- overflow.carbon

package Overflow;

fn Negate(a: u32) -> u32 = "int.unegate";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn F() {
  // -(-INT_MAX) is INT_MAX.
  Test(Negate(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  Test(-(Negate(0x7FFF_FFFF))) as Expect(0x7FFF_FFFF);
  // -(-(INT_MAX + 1)) is `INT_MAX + 1`.
  Test(Negate(Negate(0x8000_0000))) as Expect(0x8000_0000);
  Test(-(Negate(0x8000_0000))) as Expect(0x8000_0000);
}
