// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sadd_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sadd_assign.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

fn Builtin(a: i32*, b: i32) = "int.sadd_assign";

fn Call(a: i32*, b: i32) {
  Builtin(a, b);
}

// --- fail_unsized.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unsized.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Builtin(a: Core.IntLiteral()*, b: Core.IntLiteral()) = "int.sadd_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Builtin(a: Core.IntLiteral()*, b: Core.IntLiteral()) = "int.sadd_assign";

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotPtr(a: i32, b: i32) = "int.sadd_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotPtr(a: i32, b: i32) = "int.sadd_assign";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedTypes(a: i32*, b: i64) = "int.sadd_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedTypes(a: i32*, b: i64) = "int.sadd_assign";
