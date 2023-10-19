// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: file "nine_elements.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 9
// CHECK:STDOUT:   %.loc7_15.1: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %.loc7_15.2: type = ptr_type [i32; 9]
// CHECK:STDOUT:   %a: ref [i32; 9] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_23: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_29: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_32: i32 = int_literal 5
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_38: i32 = int_literal 7
// CHECK:STDOUT:   %.loc7_41: i32 = int_literal 8
// CHECK:STDOUT:   %.loc7_44: i32 = int_literal 9
// CHECK:STDOUT:   %.loc7_45.1: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc7_45.2: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = tuple_literal (%.loc7_20, %.loc7_23, %.loc7_26, %.loc7_29, %.loc7_32, %.loc7_35, %.loc7_38, %.loc7_41, %.loc7_44)
// CHECK:STDOUT:   %.loc7_45.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_45.4: ref i32 = array_index %a, %.loc7_45.3
// CHECK:STDOUT:   %.loc7_45.5: init i32 = initialize_from %.loc7_20 to %.loc7_45.4
// CHECK:STDOUT:   %.loc7_45.6: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_45.7: ref i32 = array_index %a, %.loc7_45.6
// CHECK:STDOUT:   %.loc7_45.8: init i32 = initialize_from %.loc7_23 to %.loc7_45.7
// CHECK:STDOUT:   %.loc7_45.9: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_45.10: ref i32 = array_index %a, %.loc7_45.9
// CHECK:STDOUT:   %.loc7_45.11: init i32 = initialize_from %.loc7_26 to %.loc7_45.10
// CHECK:STDOUT:   %.loc7_45.12: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_45.13: ref i32 = array_index %a, %.loc7_45.12
// CHECK:STDOUT:   %.loc7_45.14: init i32 = initialize_from %.loc7_29 to %.loc7_45.13
// CHECK:STDOUT:   %.loc7_45.15: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_45.16: ref i32 = array_index %a, %.loc7_45.15
// CHECK:STDOUT:   %.loc7_45.17: init i32 = initialize_from %.loc7_32 to %.loc7_45.16
// CHECK:STDOUT:   %.loc7_45.18: i32 = int_literal 5
// CHECK:STDOUT:   %.loc7_45.19: ref i32 = array_index %a, %.loc7_45.18
// CHECK:STDOUT:   %.loc7_45.20: init i32 = initialize_from %.loc7_35 to %.loc7_45.19
// CHECK:STDOUT:   %.loc7_45.21: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_45.22: ref i32 = array_index %a, %.loc7_45.21
// CHECK:STDOUT:   %.loc7_45.23: init i32 = initialize_from %.loc7_38 to %.loc7_45.22
// CHECK:STDOUT:   %.loc7_45.24: i32 = int_literal 7
// CHECK:STDOUT:   %.loc7_45.25: ref i32 = array_index %a, %.loc7_45.24
// CHECK:STDOUT:   %.loc7_45.26: init i32 = initialize_from %.loc7_41 to %.loc7_45.25
// CHECK:STDOUT:   %.loc7_45.27: i32 = int_literal 8
// CHECK:STDOUT:   %.loc7_45.28: ref i32 = array_index %a, %.loc7_45.27
// CHECK:STDOUT:   %.loc7_45.29: init i32 = initialize_from %.loc7_44 to %.loc7_45.28
// CHECK:STDOUT:   %.loc7_45.30: init [i32; 9] = array_init %.loc7_45.2, (%.loc7_45.5, %.loc7_45.8, %.loc7_45.11, %.loc7_45.14, %.loc7_45.17, %.loc7_45.20, %.loc7_45.23, %.loc7_45.26, %.loc7_45.29) to %a
// CHECK:STDOUT:   assign %a, %.loc7_45.30
// CHECK:STDOUT: }
