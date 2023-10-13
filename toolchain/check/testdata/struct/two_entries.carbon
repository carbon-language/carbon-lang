// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: file "two_entries.carbon" {
// CHECK:STDOUT:   %.loc7_25.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc7_25.2: type = ptr_type {.a: i32, .b: i32}
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
// CHECK:STDOUT:   %.loc8_25: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: i32} = var "y"
// CHECK:STDOUT:   %x.ref: ref {.a: i32, .b: i32} = name_reference "x", %x
// CHECK:STDOUT:   %.loc8_29.1: ref i32 = struct_access %x.ref, member0
// CHECK:STDOUT:   %.loc8_29.2: i32 = bind_value %.loc8_29.1
// CHECK:STDOUT:   %.loc8_29.3: ref i32 = struct_access %y, member0
// CHECK:STDOUT:   %.loc8_29.4: init i32 = initialize_from %.loc8_29.2 to %.loc8_29.3
// CHECK:STDOUT:   %.loc8_29.5: ref i32 = struct_access %x.ref, member1
// CHECK:STDOUT:   %.loc8_29.6: i32 = bind_value %.loc8_29.5
// CHECK:STDOUT:   %.loc8_29.7: ref i32 = struct_access %y, member1
// CHECK:STDOUT:   %.loc8_29.8: init i32 = initialize_from %.loc8_29.6 to %.loc8_29.7
// CHECK:STDOUT:   %.loc8_29.9: init {.a: i32, .b: i32} = struct_init %x.ref, (%.loc8_29.4, %.loc8_29.8)
// CHECK:STDOUT:   assign %y, %.loc8_29.9
// CHECK:STDOUT: }
