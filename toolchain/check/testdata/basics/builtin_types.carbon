// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";
var test_type: type = i32;

// CHECK:STDOUT: file "builtin_types.carbon" {
// CHECK:STDOUT:   %test_i32: ref i32 = var "test_i32"
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0
// CHECK:STDOUT:   assign %test_i32, %.loc7
// CHECK:STDOUT:   %test_f64: ref f64 = var "test_f64"
// CHECK:STDOUT:   %.loc8: f64 = real_literal 1e-1
// CHECK:STDOUT:   assign %test_f64, %.loc8
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %test_str: ref String = var "test_str"
// CHECK:STDOUT:   %.loc9: String = string_literal "Test"
// CHECK:STDOUT:   assign %test_str, %.loc9
// CHECK:STDOUT:   %test_type: ref type = var "test_type"
// CHECK:STDOUT:   assign %test_type, i32
// CHECK:STDOUT: }
