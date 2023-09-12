// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: package {
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
// CHECK:STDOUT:   %.loc7_45.3: [i32; 9] = array_value %.loc7_45.2
// CHECK:STDOUT:   assign %a, %.loc7_45.3
// CHECK:STDOUT: }
