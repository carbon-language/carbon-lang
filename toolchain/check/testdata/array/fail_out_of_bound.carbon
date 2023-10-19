// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_out_of_bound.carbon:[[@LINE+3]]:27: ERROR: Cannot initialize array of 1 element(s) from 3 initializer(s).
// CHECK:STDERR: var a: [i32; 1] = (1, 2, 3);
// CHECK:STDERR:                           ^
var a: [i32; 1] = (1, 2, 3);

// CHECK:STDOUT: file "fail_out_of_bound.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_15.1: type = array_type %.loc10_14, i32
// CHECK:STDOUT:   %.loc10_15.2: type = ptr_type [i32; 1]
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_23: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_27.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_27.2: (i32, i32, i32) = tuple_literal (%.loc10_20, %.loc10_23, %.loc10_26)
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT: }
