// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: file "assign_var.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_19: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc7_22.2: (type, type, type) = tuple_literal (%.loc7_9, %.loc7_14, %.loc7_19)
// CHECK:STDOUT:   %.loc7_22.3: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_27.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_27.2: i32 = stub_reference %.loc7_27.1
// CHECK:STDOUT:   %.loc7_30.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_30.2: i32 = stub_reference %.loc7_30.1
// CHECK:STDOUT:   %.loc7_33.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_33.2: i32 = stub_reference %.loc7_33.1
// CHECK:STDOUT:   %.loc7_34: (i32, i32, i32) = tuple_literal (%.loc7_27.2, %.loc7_30.2, %.loc7_33.2)
// CHECK:STDOUT:   assign %a, %.loc7_34
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, i32
// CHECK:STDOUT:   %b: ref [i32; 3] = var "b"
// CHECK:STDOUT:   %.loc7_5.1: ref [i32; 3] = stub_reference %b
// CHECK:STDOUT:   %.loc7_5.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_5.3: ref i32 = array_index %.loc7_5.1, %.loc7_5.2
// CHECK:STDOUT:   %.loc7_5.4: ref i32 = tuple_index %a, %.loc7_5.2
// CHECK:STDOUT:   %.loc7_5.5: i32 = bind_value %.loc7_5.4
// CHECK:STDOUT:   %.loc7_5.6: init i32 = initialize_from %.loc7_5.5 to %.loc7_5.3
// CHECK:STDOUT:   %.loc7_5.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_5.8: ref i32 = array_index %.loc7_5.1, %.loc7_5.7
// CHECK:STDOUT:   %.loc7_5.9: ref i32 = tuple_index %a, %.loc7_5.7
// CHECK:STDOUT:   %.loc7_5.10: i32 = bind_value %.loc7_5.9
// CHECK:STDOUT:   %.loc7_5.11: init i32 = initialize_from %.loc7_5.10 to %.loc7_5.8
// CHECK:STDOUT:   %.loc7_5.12: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_5.13: ref i32 = array_index %.loc7_5.1, %.loc7_5.12
// CHECK:STDOUT:   %.loc7_5.14: ref i32 = tuple_index %a, %.loc7_5.12
// CHECK:STDOUT:   %.loc7_5.15: i32 = bind_value %.loc7_5.14
// CHECK:STDOUT:   %.loc7_5.16: init i32 = initialize_from %.loc7_5.15 to %.loc7_5.13
// CHECK:STDOUT:   %.loc7_5.17: init [i32; 3] = array_init %a, (%.loc7_5.6, %.loc7_5.11, %.loc7_5.16) to %.loc7_5.1
// CHECK:STDOUT:   assign %b, %.loc7_5.17
// CHECK:STDOUT: }
