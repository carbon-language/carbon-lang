// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "member_access.carbon" {

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_25: type = struct_type {.a: f64, .b: i32}
// CHECK:STDOUT:   %x: ref {.a: f64, .b: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_35: f64 = real_literal 0e-1
// CHECK:STDOUT:   %.loc[[@LINE-4]]_33: f64 = stub_reference %.loc[[@LINE-4]]_35
// CHECK:STDOUT:   %.loc[[@LINE-5]]_45: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_43: i32 = stub_reference %.loc[[@LINE-6]]_45
// CHECK:STDOUT:   %.loc[[@LINE-7]]_46: {.a: f64, .b: i32} = struct_literal (%.loc[[@LINE-7]]_33, %.loc[[@LINE-7]]_43)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-8]]_46
// CHECK:STDOUT:   %y: ref i32 = var "y"
var y: i32 = x.b;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_15.1: ref i32 = struct_access %x, member1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15.2: i32 = bind_value %.loc[[@LINE-2]]_15.1
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-3]]_15.2
// CHECK:STDOUT:   %z: ref i32 = var "z"
// CHECK:STDOUT:   %.loc[[@LINE-5]]_5: i32 = bind_value %y
// CHECK:STDOUT:   assign %z, %.loc[[@LINE-6]]_5
// CHECK:STDOUT: }
var z: i32 = y;
