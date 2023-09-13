// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_tuple_out_of_bound_access.carbon:[[@LINE+3]]:17: Index `2` is past the end of `(i32, i32) as type`.
// CHECK:STDERR: var b: i32 = a[2];
// CHECK:STDERR:                 ^
var b: i32 = a[2];

// CHECK:STDOUT: file "fail_tuple_out_of_bound_access.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: (type, type) = tuple_literal (%.loc7_9, %.loc7_14)
// CHECK:STDOUT:   %.loc7_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_22.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_22.2: i32 = stub_reference %.loc7_22.1
// CHECK:STDOUT:   %.loc7_26.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_26.2: i32 = stub_reference %.loc7_26.1
// CHECK:STDOUT:   %.loc7_27: (i32, i32) = tuple_literal (%.loc7_22.2, %.loc7_26.2)
// CHECK:STDOUT:   assign %a, %.loc7_27
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc11_16: i32 = int_literal 2
// CHECK:STDOUT:   %.loc11_17: ref <error> = tuple_index %a, <error>
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT: }
