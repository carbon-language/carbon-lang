// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+3]]:22: Cannot implicitly convert from `{} as type` to `{.a: i32}`.
// CHECK:STDERR: var x: {.a: i32} = {};
// CHECK:STDERR:                      ^
var x: {.a: i32} = {};

// CHECK:STDOUT: file "fail_assign_empty.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc10_21.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_21.2: {} = struct_literal ()
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
