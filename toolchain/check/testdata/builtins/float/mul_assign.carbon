// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/mul_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/mul_assign.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

fn Builtin(ref a: f64, b: f64) = "float.mul_assign";

fn Call(ref a: f64, b: f64) {
  Builtin(ref a, b);
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.mul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotRef(a: f64, b: f64) = "float.mul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotRef(a: f64, b: f64) = "float.mul_assign";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.mul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedTypes(ref a: f64, b: f32) = "float.mul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedTypes(ref a: f64, b: f32) = "float.mul_assign";

// --- fail_literal.carbon

library "[[@TEST_NAME]]";

fn Literal() -> type = "float_literal.make_type";
// CHECK:STDERR: fail_literal.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.mul_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn LiteralRuntime(ref a: Literal(), b: Literal()) = "float.mul_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn LiteralRuntime(ref a: Literal(), b: Literal()) = "float.mul_assign";
