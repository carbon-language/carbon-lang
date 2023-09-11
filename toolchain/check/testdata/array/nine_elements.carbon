// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: file "nine_elements.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 9
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 9] = var "a"
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_23.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_23.2: i32 = stub_reference %.loc7_23.1
// CHECK:STDOUT:   %.loc7_26.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_26.2: i32 = stub_reference %.loc7_26.1
// CHECK:STDOUT:   %.loc7_29.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_29.2: i32 = stub_reference %.loc7_29.1
// CHECK:STDOUT:   %.loc7_32.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc7_32.2: i32 = stub_reference %.loc7_32.1
// CHECK:STDOUT:   %.loc7_35.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_35.2: i32 = stub_reference %.loc7_35.1
// CHECK:STDOUT:   %.loc7_38.1: i32 = int_literal 7
// CHECK:STDOUT:   %.loc7_38.2: i32 = stub_reference %.loc7_38.1
// CHECK:STDOUT:   %.loc7_41.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc7_41.2: i32 = stub_reference %.loc7_41.1
// CHECK:STDOUT:   %.loc7_44.1: i32 = int_literal 9
// CHECK:STDOUT:   %.loc7_44.2: i32 = stub_reference %.loc7_44.1
// CHECK:STDOUT:   %.loc7_45.1: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc7_45.2: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = tuple_literal (%.loc7_20.2, %.loc7_23.2, %.loc7_26.2, %.loc7_29.2, %.loc7_32.2, %.loc7_35.2, %.loc7_38.2, %.loc7_41.2, %.loc7_44.2)
// CHECK:STDOUT:   %.loc7_45.3: ref [i32; 9] = stub_reference %a
// CHECK:STDOUT:   %.loc7_45.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_45.5: ref i32 = array_index %.loc7_45.3, %.loc7_45.4
// CHECK:STDOUT:   %.loc7_45.6: init i32 = initialize_from %.loc7_20.2 to %.loc7_45.5
// CHECK:STDOUT:   %.loc7_45.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_45.8: ref i32 = array_index %.loc7_45.3, %.loc7_45.7
// CHECK:STDOUT:   %.loc7_45.9: init i32 = initialize_from %.loc7_23.2 to %.loc7_45.8
// CHECK:STDOUT:   %.loc7_45.10: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_45.11: ref i32 = array_index %.loc7_45.3, %.loc7_45.10
// CHECK:STDOUT:   %.loc7_45.12: init i32 = initialize_from %.loc7_26.2 to %.loc7_45.11
// CHECK:STDOUT:   %.loc7_45.13: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_45.14: ref i32 = array_index %.loc7_45.3, %.loc7_45.13
// CHECK:STDOUT:   %.loc7_45.15: init i32 = initialize_from %.loc7_29.2 to %.loc7_45.14
// CHECK:STDOUT:   %.loc7_45.16: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_45.17: ref i32 = array_index %.loc7_45.3, %.loc7_45.16
// CHECK:STDOUT:   %.loc7_45.18: init i32 = initialize_from %.loc7_32.2 to %.loc7_45.17
// CHECK:STDOUT:   %.loc7_45.19: i32 = int_literal 5
// CHECK:STDOUT:   %.loc7_45.20: ref i32 = array_index %.loc7_45.3, %.loc7_45.19
// CHECK:STDOUT:   %.loc7_45.21: init i32 = initialize_from %.loc7_35.2 to %.loc7_45.20
// CHECK:STDOUT:   %.loc7_45.22: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_45.23: ref i32 = array_index %.loc7_45.3, %.loc7_45.22
// CHECK:STDOUT:   %.loc7_45.24: init i32 = initialize_from %.loc7_38.2 to %.loc7_45.23
// CHECK:STDOUT:   %.loc7_45.25: i32 = int_literal 7
// CHECK:STDOUT:   %.loc7_45.26: ref i32 = array_index %.loc7_45.3, %.loc7_45.25
// CHECK:STDOUT:   %.loc7_45.27: init i32 = initialize_from %.loc7_41.2 to %.loc7_45.26
// CHECK:STDOUT:   %.loc7_45.28: i32 = int_literal 8
// CHECK:STDOUT:   %.loc7_45.29: ref i32 = array_index %.loc7_45.3, %.loc7_45.28
// CHECK:STDOUT:   %.loc7_45.30: init i32 = initialize_from %.loc7_44.2 to %.loc7_45.29
// CHECK:STDOUT:   %.loc7_45.31: init [i32; 9] = array_init %.loc7_45.2, (%.loc7_45.6, %.loc7_45.9, %.loc7_45.12, %.loc7_45.15, %.loc7_45.18, %.loc7_45.21, %.loc7_45.24, %.loc7_45.27, %.loc7_45.30) to %.loc7_45.3
// CHECK:STDOUT:   assign %a, %.loc7_45.31
// CHECK:STDOUT: }
