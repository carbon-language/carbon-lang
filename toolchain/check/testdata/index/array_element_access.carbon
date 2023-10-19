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
// CHECK:STDOUT:   %.loc7_15.1: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %.loc7_15.2: type = ptr_type [i32; 2]
// CHECK:STDOUT:   %a: ref [i32; 2] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_24: i32 = int_literal 24
// CHECK:STDOUT:   %.loc7_26.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_26.2: (i32, i32) = tuple_literal (%.loc7_20, %.loc7_24)
// CHECK:STDOUT:   %.loc7_26.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_26.4: ref i32 = array_index %a, %.loc7_26.3
// CHECK:STDOUT:   %.loc7_26.5: init i32 = initialize_from %.loc7_20 to %.loc7_26.4
// CHECK:STDOUT:   %.loc7_26.6: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_26.7: ref i32 = array_index %a, %.loc7_26.6
// CHECK:STDOUT:   %.loc7_26.8: init i32 = initialize_from %.loc7_24 to %.loc7_26.7
// CHECK:STDOUT:   %.loc7_26.9: init [i32; 2] = array_init %.loc7_26.2, (%.loc7_26.5, %.loc7_26.8) to %a
// CHECK:STDOUT:   assign %a, %.loc7_26.9
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1
// CHECK:STDOUT:   assign %b, %.loc8
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %a.ref.loc9: ref [i32; 2] = name_reference "a", %a
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = array_index %a.ref.loc9, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign %c, %.loc9_17.2
// CHECK:STDOUT:   %d: ref i32 = var "d"
// CHECK:STDOUT:   %a.ref.loc10: ref [i32; 2] = name_reference "a", %a
// CHECK:STDOUT:   %b.ref: ref i32 = name_reference "b", %b
// CHECK:STDOUT:   %.loc10_16: i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc10_17.1: ref i32 = array_index %a.ref.loc10, %.loc10_16
// CHECK:STDOUT:   %.loc10_17.2: i32 = bind_value %.loc10_17.1
// CHECK:STDOUT:   assign %d, %.loc10_17.2
// CHECK:STDOUT: }
