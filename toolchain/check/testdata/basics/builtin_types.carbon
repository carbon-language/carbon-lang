// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
let test_str: String = "Test";
var test_type: type = i32;

// CHECK:STDOUT: --- builtin_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: f64 = real_literal 1e-1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type String [template]
// CHECK:STDOUT:   %.4: String = string_literal "Test" [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.test_i32 = %test_i32, .test_f64 = %test_f64, .test_type = %test_type} [template]
// CHECK:STDOUT:   %test_i32.var: ref i32 = var test_i32
// CHECK:STDOUT:   %test_i32: ref i32 = bind_name test_i32, %test_i32.var
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %test_i32.var, %.loc7
// CHECK:STDOUT:   %test_f64.var: ref f64 = var test_f64
// CHECK:STDOUT:   %test_f64: ref f64 = bind_name test_f64, %test_f64.var
// CHECK:STDOUT:   %.loc8: f64 = real_literal 1e-1 [template = constants.%.2]
// CHECK:STDOUT:   assign %test_f64.var, %.loc8
// CHECK:STDOUT:   %.loc9: String = string_literal "Test" [template = constants.%.4]
// CHECK:STDOUT:   %test_str: String = bind_name test_str, %.loc9
// CHECK:STDOUT:   %test_type.var: ref type = var test_type
// CHECK:STDOUT:   %test_type: ref type = bind_name test_type, %test_type.var
// CHECK:STDOUT:   assign %test_type.var, i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
