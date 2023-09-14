// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (12,);
// CHECK:STDERR: fail_array_non_int_indexing.carbon:[[@LINE+3]]:16: Cannot implicitly convert from `f64` to `i32`.
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                ^
var b: i32 = a[2.6];

// CHECK:STDOUT: file "fail_array_non_int_indexing.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_23.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_23.2: (i32,) = tuple_literal (%.loc7_20.2)
// CHECK:STDOUT:   %.loc7_23.3: ref [i32; 1] = stub_reference %a
// CHECK:STDOUT:   %.loc7_23.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_23.5: ref i32 = array_index %.loc7_23.3, %.loc7_23.4
// CHECK:STDOUT:   %.loc7_23.6: init i32 = initialize_from %.loc7_20.2 to %.loc7_23.5
// CHECK:STDOUT:   %.loc7_23.7: init [i32; 1] = array_init %.loc7_23.2, (%.loc7_23.6) to %.loc7_23.3
// CHECK:STDOUT:   assign %a, %.loc7_23.7
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc11_16: f64 = real_literal 26e-1
// CHECK:STDOUT:   %.loc11_19.1: ref i32 = array_index %a, <error>
// CHECK:STDOUT:   %.loc11_19.2: i32 = bind_value %.loc11_19.1
// CHECK:STDOUT:   assign %b, %.loc11_19.2
// CHECK:STDOUT: }
