// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_field_type_mismatch.carbon:[[@LINE+3]]:20: ERROR: Missing value for field `a` in struct initialization.
// CHECK:STDERR: var x: {.a: i32} = {.b = 1.0};
// CHECK:STDERR:                    ^~~~~~~~~~
var x: {.a: i32} = {.b = 1.0};

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10: type = struct_type {.b: f64}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_field_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_26: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc10_29: {.b: f64} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
