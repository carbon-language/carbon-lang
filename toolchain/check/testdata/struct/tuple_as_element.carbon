// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
var y: {.a: i32, .b: (i32,)} = x;

// CHECK:STDOUT: file "tuple_as_element.carbon" {
// CHECK:STDOUT:   %.loc7_23: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_27.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_27.2: (type,) = tuple_literal (%.loc7_23)
// CHECK:STDOUT:   %.loc7_27.3: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_28: type = struct_type {.a: i32, .b: (i32,)}
// CHECK:STDOUT:   %x: ref {.a: i32, .b: (i32,)} = var "x"
// CHECK:STDOUT:   %.loc7_38: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_36: i32 = stub_reference %.loc7_38
// CHECK:STDOUT:   %.loc7_47.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_47.2: i32 = stub_reference %.loc7_47.1
// CHECK:STDOUT:   %.loc7_49: (i32,) = tuple_literal (%.loc7_47.2)
// CHECK:STDOUT:   %.loc7_44: (i32,) = stub_reference %.loc7_49
// CHECK:STDOUT:   %.loc7_50: {.a: i32, .b: (i32,)} = struct_literal (%.loc7_36, %.loc7_44)
// CHECK:STDOUT:   %.loc7_51.1: ref i32 = struct_access %x, member0
// CHECK:STDOUT:   %.loc7_51.2: init i32 = initialize_from %.loc7_36 to %.loc7_51.1
// CHECK:STDOUT:   %.loc7_51.3: ref (i32,) = struct_access %x, member1
// CHECK:STDOUT:   %.loc7_51.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_51.5: ref i32 = tuple_index %.loc7_51.3, %.loc7_51.4
// CHECK:STDOUT:   %.loc7_51.6: init (i32,) = tuple_init %.loc7_49, (%.loc7_47.2)
// CHECK:STDOUT:   %.loc7_51.7: init (i32,) = initialize_from %.loc7_51.6 to %.loc7_51.3
// CHECK:STDOUT:   %.loc7_51.8: init {.a: i32, .b: (i32,)} = struct_init %.loc7_50, (%.loc7_51.2, %.loc7_51.7)
// CHECK:STDOUT:   assign %x, %.loc7_51.8
// CHECK:STDOUT:   %.loc8_23: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_27: (type,) = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %.loc8_28: type = struct_type {.a: i32, .b: (i32,)}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: (i32,)} = var "y"
// CHECK:STDOUT:   %.loc7_5: {.a: i32, .b: (i32,)} = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc7_5
// CHECK:STDOUT: }
