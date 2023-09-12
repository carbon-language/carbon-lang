// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "tuple_as_element.carbon" {

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_23: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_27.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc[[@LINE-3]]_27.2: (type,) = tuple_literal (%.loc[[@LINE-3]]_23)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_27.3: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_28: type = struct_type {.a: i32, .b: (i32,)}
// CHECK:STDOUT:   %x: ref {.a: i32, .b: (i32,)} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-7]]_38: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_36: i32 = stub_reference %.loc[[@LINE-8]]_38
// CHECK:STDOUT:   %.loc[[@LINE-9]]_47.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-10]]_47.2: i32 = stub_reference %.loc[[@LINE-10]]_47.1
// CHECK:STDOUT:   %.loc[[@LINE-11]]_49: (i32,) = tuple_literal (%.loc[[@LINE-11]]_47.2)
// CHECK:STDOUT:   %.loc[[@LINE-12]]_44: (i32,) = stub_reference %.loc[[@LINE-12]]_49
// CHECK:STDOUT:   %.loc[[@LINE-13]]_50: {.a: i32, .b: (i32,)} = struct_literal (%.loc[[@LINE-13]]_36, %.loc[[@LINE-13]]_44)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-14]]_50
var y: {.a: i32, .b: (i32,)} = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_23: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_27: (type,) = tuple_literal (%.loc[[@LINE-2]]_23)
// CHECK:STDOUT:   %.loc[[@LINE-3]]_28: type = struct_type {.a: i32, .b: (i32,)}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: (i32,)} = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-20]]_5: {.a: i32, .b: (i32,)} = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-21]]_5
// CHECK:STDOUT: }
