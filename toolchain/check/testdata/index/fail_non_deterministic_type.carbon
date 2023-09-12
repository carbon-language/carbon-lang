// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_non_deterministic_type.carbon" {

var a: (i32, i32) = (2, 3);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_17.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_9, %.loc[[@LINE-4]]_14)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-7]]_22.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-8]]_22.2: i32 = stub_reference %.loc[[@LINE-8]]_22.1
// CHECK:STDOUT:   %.loc[[@LINE-9]]_25.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-10]]_25.2: i32 = stub_reference %.loc[[@LINE-10]]_25.1
// CHECK:STDOUT:   %.loc[[@LINE-11]]_26: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_22.2, %.loc[[@LINE-11]]_25.2)
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-12]]_26
// CHECK:STDOUT:   %b: ref i32 = var "b"
var b: i32 = 0;
// CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 0
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-2]]
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+3]]:17: Tuples indices must be integer literals.
// CHECK:STDERR: var c: i32 = a[b];
// CHECK:STDERR:                 ^
var c: i32 = a[b];
// CHECK:STDOUT:   %.loc[[@LINE-1]]: ref <error> = tuple_index %a, <error>
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT: }
