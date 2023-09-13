// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: file "member_access.carbon" {
// CHECK:STDOUT:   %.loc7_25: type = struct_type {.a: f64, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: f64, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc7_35: f64 = real_literal 0e-1
// CHECK:STDOUT:   %.loc7_33: f64 = stub_reference %.loc7_35
// CHECK:STDOUT:   %.loc7_45: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_43: i32 = stub_reference %.loc7_45
// CHECK:STDOUT:   %.loc7_46: {.a: f64, .b: i32} = struct_literal (%.loc7_33, %.loc7_43)
// CHECK:STDOUT:   assign %x, %.loc7_46
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %.loc8_15.1: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc8_15.2: i32 = bind_value %.loc8_15.1
// CHECK:STDOUT:   assign %y, %.loc8_15.2
// CHECK:STDOUT:   %z: ref i32 = var "z"
// CHECK:STDOUT:   %.loc8_5: i32 = bind_value %y
// CHECK:STDOUT:   assign %z, %.loc8_5
// CHECK:STDOUT: }
