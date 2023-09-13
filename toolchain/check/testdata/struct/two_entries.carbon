// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: file "two_entries.carbon" {
// CHECK:STDOUT:   %.loc7_25: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: i32, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_33: i32 = stub_reference %.loc7_35
// CHECK:STDOUT:   %.loc7_43: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_41: i32 = stub_reference %.loc7_43
// CHECK:STDOUT:   %.loc7_44: {.a: i32, .b: i32} = struct_literal (%.loc7_33, %.loc7_41)
// CHECK:STDOUT:   assign %x, %.loc7_44
// CHECK:STDOUT:   %.loc8: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: i32} = var "y"
// CHECK:STDOUT:   %.loc7_5: {.a: i32, .b: i32} = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc7_5
// CHECK:STDOUT: }
