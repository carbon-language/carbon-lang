// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+15]]:10: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+12]]:10: Member with the same name here.
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+9]]:22: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR:                      ^
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+6]]:22: Member with the same name here.
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR:                      ^
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+3]]:29: ERROR: Cannot implicitly convert from `type` to `{.a: i32}`.
// CHECK:STDERR: var x: {.a: i32} = {.a: i32};
// CHECK:STDERR:                             ^
var x: {.a: i32} = {.a: i32};

// CHECK:STDOUT: file "fail_type_assign.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_28: type = struct_type {.a: i32}
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
