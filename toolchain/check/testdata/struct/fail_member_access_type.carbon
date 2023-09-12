// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_member_access_type.carbon" {

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: type = struct_type {.a: f64}
// CHECK:STDOUT:   %x: ref {.a: f64} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_26: f64 = real_literal 40e-1
// CHECK:STDOUT:   %.loc[[@LINE-4]]_24: f64 = stub_reference %.loc[[@LINE-4]]_26
// CHECK:STDOUT:   %.loc[[@LINE-5]]_29: {.a: f64} = struct_literal (%.loc[[@LINE-5]]_24)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-6]]_29
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+3]]:15: Type `{.a: f64}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:               ^
var y: i32 = x.b;
