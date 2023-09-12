// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "assign_var.carbon" {

var a: (i32, i32, i32) = (1, 2, 3);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_19: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-4]]_22.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_22.2: (type, type, type) = tuple_literal (%.loc[[@LINE-5]]_9, %.loc[[@LINE-5]]_14, %.loc[[@LINE-5]]_19)
// CHECK:STDOUT:   %.loc[[@LINE-6]]_22.3: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32, i32) = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-8]]_27.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-9]]_27.2: i32 = stub_reference %.loc[[@LINE-9]]_27.1
// CHECK:STDOUT:   %.loc[[@LINE-10]]_30.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-11]]_30.2: i32 = stub_reference %.loc[[@LINE-11]]_30.1
// CHECK:STDOUT:   %.loc[[@LINE-12]]_33.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-13]]_33.2: i32 = stub_reference %.loc[[@LINE-13]]_33.1
// CHECK:STDOUT:   %.loc[[@LINE-14]]_34: (i32, i32, i32) = tuple_literal (%.loc[[@LINE-14]]_27.2, %.loc[[@LINE-14]]_30.2, %.loc[[@LINE-14]]_33.2)
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-15]]_34
var b: [i32; 3] = a;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %b: ref [i32; 3] = var "b"
// CHECK:STDOUT:   %.loc[[@LINE-20]]_5.1: (i32, i32, i32) = bind_value %a
// CHECK:STDOUT:   %.loc[[@LINE-21]]_5.2: [i32; 3] = array_value %.loc[[@LINE-21]]_5.1
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-22]]_5.2
// CHECK:STDOUT: }
