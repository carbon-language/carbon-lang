// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+3]]:15: Type `{.a: f64}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:               ^
var y: i32 = x.b;

// CHECK:STDOUT: file "fail_member_access_type.carbon" {
// CHECK:STDOUT:   %.loc7_16: type = struct_type {.a: f64}
// CHECK:STDOUT:   %x: ref {.a: f64} = var "x"
// CHECK:STDOUT:   %.loc7_26: f64 = real_literal 40e-1
// CHECK:STDOUT:   %.loc7_24: f64 = stub_reference %.loc7_26
// CHECK:STDOUT:   %.loc7_29: {.a: f64} = struct_literal (%.loc7_24)
// CHECK:STDOUT:   assign %x, %.loc7_29
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }
