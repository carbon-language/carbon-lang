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
// CHECK:STDOUT:   %.loc7_43: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_44.1: {.a: i32, .b: i32} = struct_literal (%.loc7_35, %.loc7_43)
// CHECK:STDOUT:   %.loc7_44.2: ref i32 = struct_access %x, member0
// CHECK:STDOUT:   %.loc7_44.3: init i32 = initialize_from %.loc7_35 to %.loc7_44.2
// CHECK:STDOUT:   %.loc7_44.4: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc7_44.5: init i32 = initialize_from %.loc7_43 to %.loc7_44.4
// CHECK:STDOUT:   %.loc7_44.6: init {.a: i32, .b: i32} = struct_init %.loc7_44.1, (%.loc7_44.3, %.loc7_44.5)
// CHECK:STDOUT:   assign %x, %.loc7_44.6
// CHECK:STDOUT:   %.loc8: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: i32} = var "y"
// CHECK:STDOUT:   %.loc7_5.1: ref i32 = struct_access %x, member0
// CHECK:STDOUT:   %.loc7_5.2: i32 = bind_value %.loc7_5.1
// CHECK:STDOUT:   %.loc7_5.3: ref i32 = struct_access %y, member0
// CHECK:STDOUT:   %.loc7_5.4: init i32 = initialize_from %.loc7_5.2 to %.loc7_5.3
// CHECK:STDOUT:   %.loc7_5.5: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc7_5.6: i32 = bind_value %.loc7_5.5
// CHECK:STDOUT:   %.loc7_5.7: ref i32 = struct_access %y, member1
// CHECK:STDOUT:   %.loc7_5.8: init i32 = initialize_from %.loc7_5.6 to %.loc7_5.7
// CHECK:STDOUT:   %.loc7_5.9: init {.a: i32, .b: i32} = struct_init %x, (%.loc7_5.4, %.loc7_5.8)
// CHECK:STDOUT:   assign %y, %.loc7_5.9
// CHECK:STDOUT: }
