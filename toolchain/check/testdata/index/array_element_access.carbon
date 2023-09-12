// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "array_element_access.carbon" {

var a: [i32; 2] = (12, 24);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 2] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_24.1: i32 = int_literal 24
// CHECK:STDOUT:   %.loc[[@LINE-7]]_24.2: i32 = stub_reference %.loc[[@LINE-7]]_24.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_26.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-9]]_26.2: (i32, i32) = tuple_literal (%.loc[[@LINE-9]]_20.2, %.loc[[@LINE-9]]_24.2)
// CHECK:STDOUT:   %.loc[[@LINE-10]]_26.3: [i32; 2] = array_value %.loc[[@LINE-10]]_26.2
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-11]]_26.3
// CHECK:STDOUT:   %b: ref i32 = var "b"
var b: i32 = 1;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 1
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-2]]_14
// CHECK:STDOUT:   %c: ref i32 = var "c"
var c: i32 = a[0];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc[[@LINE-2]]_17.1: ref i32 = array_index %a, %.loc[[@LINE-2]]_16
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17.2: i32 = bind_value %.loc[[@LINE-3]]_17.1
// CHECK:STDOUT:   assign %c, %.loc[[@LINE-4]]_17.2
// CHECK:STDOUT:   %d: ref i32 = var "d"
// CHECK:STDOUT:   %.loc[[@LINE-10]]_5: i32 = bind_value %b
var d: i32 = a[b];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_17.1: ref i32 = array_index %a, %.loc[[@LINE-12]]_5
// CHECK:STDOUT:   %.loc[[@LINE-2]]_17.2: i32 = bind_value %.loc[[@LINE-2]]_17.1
// CHECK:STDOUT:   assign %d, %.loc[[@LINE-3]]_17.2
// CHECK:STDOUT: }
