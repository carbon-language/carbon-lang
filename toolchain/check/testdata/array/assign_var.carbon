// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: file "assign_var.carbon" {
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc7_22.2: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_22.3: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_22.4: type = ptr_type (i32, i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_27: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_33: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_34.1: (i32, i32, i32) = tuple_literal (%.loc7_27, %.loc7_30, %.loc7_33)
// CHECK:STDOUT:   %.loc7_34.2: ref i32 = tuple_access %a, member0
// CHECK:STDOUT:   %.loc7_34.3: init i32 = initialize_from %.loc7_27 to %.loc7_34.2
// CHECK:STDOUT:   %.loc7_34.4: ref i32 = tuple_access %a, member1
// CHECK:STDOUT:   %.loc7_34.5: init i32 = initialize_from %.loc7_30 to %.loc7_34.4
// CHECK:STDOUT:   %.loc7_34.6: ref i32 = tuple_access %a, member2
// CHECK:STDOUT:   %.loc7_34.7: init i32 = initialize_from %.loc7_33 to %.loc7_34.6
// CHECK:STDOUT:   %.loc7_34.8: init (i32, i32, i32) = tuple_init %.loc7_34.1, (%.loc7_34.3, %.loc7_34.5, %.loc7_34.7)
// CHECK:STDOUT:   assign %a, %.loc7_34.8
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc8_15.1: type = array_type %.loc8_14, i32
// CHECK:STDOUT:   %.loc8_15.2: type = ptr_type [i32; 3]
// CHECK:STDOUT:   %b: ref [i32; 3] = var "b"
// CHECK:STDOUT:   %a.ref: ref (i32, i32, i32) = name_reference "a", %a
// CHECK:STDOUT:   %.loc8_19.1: ref i32 = tuple_access %a.ref, member0
// CHECK:STDOUT:   %.loc8_19.2: i32 = bind_value %.loc8_19.1
// CHECK:STDOUT:   %.loc8_19.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc8_19.4: ref i32 = array_index %b, %.loc8_19.3
// CHECK:STDOUT:   %.loc8_19.5: init i32 = initialize_from %.loc8_19.2 to %.loc8_19.4
// CHECK:STDOUT:   %.loc8_19.6: ref i32 = tuple_access %a.ref, member1
// CHECK:STDOUT:   %.loc8_19.7: i32 = bind_value %.loc8_19.6
// CHECK:STDOUT:   %.loc8_19.8: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_19.9: ref i32 = array_index %b, %.loc8_19.8
// CHECK:STDOUT:   %.loc8_19.10: init i32 = initialize_from %.loc8_19.7 to %.loc8_19.9
// CHECK:STDOUT:   %.loc8_19.11: ref i32 = tuple_access %a.ref, member2
// CHECK:STDOUT:   %.loc8_19.12: i32 = bind_value %.loc8_19.11
// CHECK:STDOUT:   %.loc8_19.13: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_19.14: ref i32 = array_index %b, %.loc8_19.13
// CHECK:STDOUT:   %.loc8_19.15: init i32 = initialize_from %.loc8_19.12 to %.loc8_19.14
// CHECK:STDOUT:   %.loc8_19.16: init [i32; 3] = array_init %a.ref, (%.loc8_19.5, %.loc8_19.10, %.loc8_19.15) to %b
// CHECK:STDOUT:   assign %b, %.loc8_19.16
// CHECK:STDOUT: }
