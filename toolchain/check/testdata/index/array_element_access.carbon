// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: file "array_element_access.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 2] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_24: i32 = int_literal 24
// CHECK:STDOUT:   %.loc7_26.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_26.2: (i32, i32) = tuple_literal (%.loc7_20, %.loc7_24)
// CHECK:STDOUT:   %.loc7_26.3: ref [i32; 2] = stub_reference %a
// CHECK:STDOUT:   %.loc7_26.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_26.5: ref i32 = array_index %.loc7_26.3, %.loc7_26.4
// CHECK:STDOUT:   %.loc7_26.6: init i32 = initialize_from %.loc7_20 to %.loc7_26.5
// CHECK:STDOUT:   %.loc7_26.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_26.8: ref i32 = array_index %.loc7_26.3, %.loc7_26.7
// CHECK:STDOUT:   %.loc7_26.9: init i32 = initialize_from %.loc7_24 to %.loc7_26.8
// CHECK:STDOUT:   %.loc7_26.10: init [i32; 2] = array_init %.loc7_26.2, (%.loc7_26.6, %.loc7_26.9) to %.loc7_26.3
// CHECK:STDOUT:   assign %a, %.loc7_26.10
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 1
// CHECK:STDOUT:   assign %b, %.loc8_14
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = array_index %a, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign %c, %.loc9_17.2
// CHECK:STDOUT:   %d: ref i32 = var "d"
// CHECK:STDOUT:   %.loc8_5: i32 = bind_value %b
// CHECK:STDOUT:   %.loc10_17.1: ref i32 = array_index %a, %.loc8_5
// CHECK:STDOUT:   %.loc10_17.2: i32 = bind_value %.loc10_17.1
// CHECK:STDOUT:   assign %d, %.loc10_17.2
// CHECK:STDOUT: }
