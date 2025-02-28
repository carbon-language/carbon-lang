// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/and.carbon

// --- int_and.carbon

library "[[@TEST_NAME]]";

fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: array(i32, And(12, 10));
let arr_p: array(i32, 8)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return And(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(And(1, 2)) as Expect(0);
  Test(And(12, 10)) as Expect(8);

  Test(And(1, -1)) as Expect(1);
  Test(And(-2, -3)) as Expect(-4);
  // Ensure the sign bit is treated properly even for 64-bit numbers.
  Test(And(0x7FFF_FFFF_FFFF_FFFF, -3)) as Expect(0x7FFF_FFFF_FFFF_FFFD);
  Test(And(0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";

fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

fn F(a: Core.IntLiteral()) -> Core.IntLiteral() {
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+7]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return AndLit(5, a);
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return AndLit(5, a);
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// Heterogeneous "and" is not supported.
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd1(a: i32, b: Core.IntLiteral()) -> i32 = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd2(a: Core.IntLiteral(), b: i32) -> i32 = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd3(a: i32, b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.and" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedAnd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.and";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedAnd4(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.and";

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

fn Test(n: Core.IntLiteral()) {
  // OK
  And(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(n, 1);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  And(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(1, n);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  And(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   And(n, n);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn And(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  And(n, n);
}
