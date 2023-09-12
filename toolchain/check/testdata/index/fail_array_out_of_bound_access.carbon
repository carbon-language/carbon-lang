// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_array_out_of_bound_access.carbon" {

var a: [i32; 1] = (12,);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_23.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc[[@LINE-7]]_23.2: (i32,) = tuple_literal (%.loc[[@LINE-7]]_20.2)
// CHECK:STDOUT:   %.loc[[@LINE-8]]_23.3: [i32; 1] = array_value %.loc[[@LINE-8]]_23.2
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-9]]_23.3
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDERR: fail_array_out_of_bound_access.carbon:[[@LINE+3]]:17: Index `2` is past the end of `[i32; 1]`.
// CHECK:STDERR: var b: i32 = a[2];
// CHECK:STDERR:                 ^
var b: i32 = a[2];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-2]]_17.1: ref i32 = array_index %a, <error>
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17.2: i32 = bind_value %.loc[[@LINE-3]]_17.1
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-4]]_17.2
// CHECK:STDOUT: }
