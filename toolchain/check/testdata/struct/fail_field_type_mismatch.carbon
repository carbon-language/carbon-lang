// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_field_type_mismatch.carbon:[[@LINE+3]]:29: ERROR: Mismatched names for field 1 in struct initialization: source has field name `b`, destination has field name `a`.
// CHECK:STDERR: var x: {.a: i32} = {.b = 1.0};
// CHECK:STDERR:                             ^
var x: {.a: i32} = {.b = 1.0};

// CHECK:STDOUT: file "fail_field_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc10_26: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc10_29.1: type = struct_type {.b: f64}
// CHECK:STDOUT:   %.loc10_29.2: {.b: f64} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
