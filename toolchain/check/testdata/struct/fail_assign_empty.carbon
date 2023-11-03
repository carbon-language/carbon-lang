// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+3]]:21: ERROR: Cannot initialize struct with 1 field(s) from struct with 0 field(s).
// CHECK:STDERR: var x: {.a: i32} = {};
// CHECK:STDERR:                     ^
var x: {.a: i32} = {};

// CHECK:STDOUT: file "fail_assign_empty.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name "x", %x.var
// CHECK:STDOUT:   %.loc10_21.1: type = struct_type {}
// CHECK:STDOUT:   %.loc10_21.2: {} = struct_literal ()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
