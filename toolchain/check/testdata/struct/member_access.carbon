// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: file "member_access.carbon" {
// CHECK:STDOUT:   %.loc7_25.1: type = struct_type {.a: f64, .b: i32}
// CHECK:STDOUT:   %.loc7_25.2: type = ptr_type {.a: f64, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: f64, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc7_35: f64 = real_literal 0e-1
// CHECK:STDOUT:   %.loc7_45: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_46.1: {.a: f64, .b: i32} = struct_literal (%.loc7_35, %.loc7_45)
// CHECK:STDOUT:   %.loc7_46.2: ref f64 = struct_access %x, member0
// CHECK:STDOUT:   %.loc7_46.3: init f64 = initialize_from %.loc7_35 to %.loc7_46.2
// CHECK:STDOUT:   %.loc7_46.4: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc7_46.5: init i32 = initialize_from %.loc7_45 to %.loc7_46.4
// CHECK:STDOUT:   %.loc7_46.6: init {.a: f64, .b: i32} = struct_init %.loc7_46.1, (%.loc7_46.3, %.loc7_46.5)
// CHECK:STDOUT:   assign %x, %.loc7_46.6
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %x.ref: ref {.a: f64, .b: i32} = name_reference "x", %x
// CHECK:STDOUT:   %.loc8_15.1: ref i32 = struct_access %x.ref, member1
// CHECK:STDOUT:   %.loc8_15.2: i32 = bind_value %.loc8_15.1
// CHECK:STDOUT:   assign %y, %.loc8_15.2
// CHECK:STDOUT:   %z: ref i32 = var "z"
// CHECK:STDOUT:   %y.ref: ref i32 = name_reference "y", %y
// CHECK:STDOUT:   %.loc9: i32 = bind_value %y.ref
// CHECK:STDOUT:   assign %z, %.loc9
// CHECK:STDOUT: }
