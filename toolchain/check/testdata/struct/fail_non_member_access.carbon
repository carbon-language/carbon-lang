// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32} = {.a = 4};
// CHECK:STDERR: fail_non_member_access.carbon:[[@LINE+3]]:15: ERROR: Type `{.a: i32}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:               ^
var y: i32 = x.b;

// CHECK:STDOUT: file "fail_non_member_access.carbon" {
// CHECK:STDOUT:   %.loc7_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_27.1: {.a: i32} = struct_literal (%.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: init {.a: i32} = struct_init %.loc7_27.1, (%.loc7_26)
// CHECK:STDOUT:   assign %x, %.loc7_27.2
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %x.ref: ref {.a: i32} = name_reference "x", %x
// CHECK:STDOUT:   assign %y, <error>
// CHECK:STDOUT: }
