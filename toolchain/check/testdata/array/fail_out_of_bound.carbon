// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_out_of_bound.carbon" {

// CHECK:STDERR: fail_out_of_bound.carbon:[[@LINE+3]]:28: Cannot implicitly convert from `(i32, i32, i32) as type` to `[i32; 1]`.
// CHECK:STDERR: var a: [i32; 1] = (1, 2, 3);
// CHECK:STDERR:                            ^
var a: [i32; 1] = (1, 2, 3);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_23.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-7]]_23.2: i32 = stub_reference %.loc[[@LINE-7]]_23.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_26.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-9]]_26.2: i32 = stub_reference %.loc[[@LINE-9]]_26.1
// CHECK:STDOUT:   %.loc[[@LINE-10]]_27.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-11]]_27.2: (i32, i32, i32) = tuple_literal (%.loc[[@LINE-11]]_20.2, %.loc[[@LINE-11]]_23.2, %.loc[[@LINE-11]]_26.2)
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT: }
