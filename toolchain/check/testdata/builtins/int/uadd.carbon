// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/uadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/uadd.carbon

// --- int_add.carbon

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

var arr: array(i32, Add(1, 2));
let arr_p: array(i32, 3)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.uadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.uadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.uadd" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.uadd";
fn JustRight(a: i32, b: i32) -> i32 = "int.uadd";

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
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.uadd";
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

// --- overflow.carbon

package Overflow;

fn Add(a: i32, b: i32) -> i32 = "int.uadd";

// Overflow is OK.
let a: i32 = Add(0x7FFFFFFF, 0);
let b: i32 = Add(0x7FFFFFFF, 1);
