// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/div_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/div_assign.carbon

// --- call.carbon

library "call";

fn Builtin(a: f64*, b: f64) = "float.div_assign";

fn Call(a: f64*, b: f64) {
  Builtin(a, b);
}

// --- fail_bad_decl.carbon

library "fail_bad_decl";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.div_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn NotPtr(a: f64, b: f64) = "float.div_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotPtr(a: f64, b: f64) = "float.div_assign";

// CHECK:STDERR: min_prelude/parts/float.carbon:11:9: error: bit width must be 64 [CompileTimeFloatBitWidth]
// CHECK:STDERR:   adapt MakeFloat(N);
// CHECK:STDERR:         ^~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+8]]:24: note: in `f32` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn MixedTypes(a: f64*, b: f32) = "float.div_assign";
// CHECK:STDERR:                        ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.div_assign" [InvalidBuiltinSignature]
// CHECK:STDERR: fn MixedTypes(a: f64*, b: f32) = "float.div_assign";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn MixedTypes(a: f64*, b: f32) = "float.div_assign";
