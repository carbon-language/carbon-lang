// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (12,);
// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+3]]:35: ERROR: Index `295147905179352825855` is past the end of `[i32; 1]`.
// CHECK:STDERR: var b: i32 = a[0xFFFFFFFFFFFFFFFFF];
// CHECK:STDERR:                                   ^
var b: i32 = a[0xFFFFFFFFFFFFFFFFF];

// CHECK:STDOUT: file "fail_array_large_index.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_15.1: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %.loc7_15.2: type = ptr_type [i32; 1]
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_23.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_23.2: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_23.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_23.4: ref i32 = array_index %a, %.loc7_23.3
// CHECK:STDOUT:   %.loc7_23.5: init i32 = initialize_from %.loc7_20 to %.loc7_23.4
// CHECK:STDOUT:   %.loc7_23.6: init [i32; 1] = array_init %.loc7_23.2, (%.loc7_23.5) to %a
// CHECK:STDOUT:   assign %a, %.loc7_23.6
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %a.ref: ref [i32; 1] = name_reference "a", %a
// CHECK:STDOUT:   %.loc11_16: i32 = int_literal 295147905179352825855
// CHECK:STDOUT:   %.loc11_35.1: ref i32 = array_index %a.ref, <error>
// CHECK:STDOUT:   %.loc11_35.2: i32 = bind_value %.loc11_35.1
// CHECK:STDOUT:   assign %b, %.loc11_35.2
// CHECK:STDOUT: }
