// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_non_member_access.carbon" {

var x: {.a: i32} = {.a = 4};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_26: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-4]]_24: i32 = stub_reference %.loc[[@LINE-4]]_26
// CHECK:STDOUT:   %.loc[[@LINE-5]]_27: {.a: i32} = struct_literal (%.loc[[@LINE-5]]_24)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-6]]_27
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }
// CHECK:STDERR: fail_non_member_access.carbon:[[@LINE+3]]:15: Type `{.a: i32}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:               ^
var y: i32 = x.b;
