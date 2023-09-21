// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_tuple_non_int_indexing.carbon:[[@LINE+3]]:19: Tuples indices must be integer literals.
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                   ^
var b: i32 = a[2.6];

// CHECK:STDOUT: file "fail_tuple_non_int_indexing.carbon" {
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_27: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_28.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_28.2: ref i32 = tuple_index %a, %.loc7_28.1
// CHECK:STDOUT:   %.loc7_28.3: init i32 = initialize_from %.loc7_22 to %.loc7_28.2
// CHECK:STDOUT:   %.loc7_28.4: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_28.5: ref i32 = tuple_index %a, %.loc7_28.4
// CHECK:STDOUT:   %.loc7_28.6: init i32 = initialize_from %.loc7_26 to %.loc7_28.5
// CHECK:STDOUT:   %.loc7_28.7: init (i32, i32) = tuple_init %.loc7_27, (%.loc7_28.3, %.loc7_28.6)
// CHECK:STDOUT:   assign %a, %.loc7_28.7
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc11_16: f64 = real_literal 26e-1
// CHECK:STDOUT:   %.loc11_19: ref <error> = tuple_index %a, <error>
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT: }
