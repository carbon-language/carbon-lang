// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_field_name_mismatch.carbon:[[@LINE+3]]:27: ERROR: Mismatched names for field 1 in struct initialization: source has field name `b`, destination has field name `a`.
// CHECK:STDERR: var x: {.a: i32} = {.b = 1};
// CHECK:STDERR:                           ^
var x: {.a: i32} = {.b = 1};

// CHECK:STDOUT: file "fail_field_name_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_27.1: type = struct_type {.b: i32}
// CHECK:STDOUT:   %.loc10_27.2: {.b: i32} = struct_literal (%.loc10_26)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
