// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "two_entries.carbon" {

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_25: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: i32, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-4]]_33: i32 = stub_reference %.loc[[@LINE-4]]_35
// CHECK:STDOUT:   %.loc[[@LINE-5]]_43: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-6]]_41: i32 = stub_reference %.loc[[@LINE-6]]_43
// CHECK:STDOUT:   %.loc[[@LINE-7]]_44: {.a: i32, .b: i32} = struct_literal (%.loc[[@LINE-7]]_33, %.loc[[@LINE-7]]_41)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-8]]_44
var y: {.a: i32, .b: i32} = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %y: ref {.a: i32, .b: i32} = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-12]]_5: {.a: i32, .b: i32} = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-13]]_5
// CHECK:STDOUT: }
