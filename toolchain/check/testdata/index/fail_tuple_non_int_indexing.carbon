// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_tuple_non_int_indexing.carbon" {

var a: (i32, i32) = (12, 6);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_17.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_9, %.loc[[@LINE-4]]_14)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-7]]_22.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc[[@LINE-8]]_22.2: i32 = stub_reference %.loc[[@LINE-8]]_22.1
// CHECK:STDOUT:   %.loc[[@LINE-9]]_26.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc[[@LINE-10]]_26.2: i32 = stub_reference %.loc[[@LINE-10]]_26.1
// CHECK:STDOUT:   %.loc[[@LINE-11]]_27: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_22.2, %.loc[[@LINE-11]]_26.2)
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-12]]_27
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDERR: fail_tuple_non_int_indexing.carbon:[[@LINE+3]]:19: Tuples indices must be integer literals.
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                   ^
var b: i32 = a[2.6];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: f64 = real_literal 26e-1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_19: ref <error> = tuple_index %a, <error>
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT: }
