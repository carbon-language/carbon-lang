// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_field_type_mismatch.carbon" {

// CHECK:STDERR: fail_field_type_mismatch.carbon:[[@LINE+3]]:30: Cannot implicitly convert from `{.b: f64}` to `{.a: i32}`.
// CHECK:STDERR: var x: {.a: i32} = {.b = 1.0};
// CHECK:STDERR:                              ^
var x: {.a: i32} = {.b = 1.0};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_26: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc[[@LINE-4]]_24: f64 = stub_reference %.loc[[@LINE-4]]_26
// CHECK:STDOUT:   %.loc[[@LINE-5]]_29.1: type = struct_type {.b: f64}
// CHECK:STDOUT:   %.loc[[@LINE-6]]_29.2: {.b: f64} = struct_literal (%.loc[[@LINE-6]]_24)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
