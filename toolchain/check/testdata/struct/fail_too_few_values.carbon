// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_too_few_values.carbon:[[@LINE+3]]:29: ERROR: Cannot initialize struct with 2 field(s) from struct with 1 field(s).
// CHECK:STDERR: var x: {.a: i32, .b: i32} = {.a = 1};
// CHECK:STDERR:                             ^~~~~~~~
var x: {.a: i32, .b: i32} = {.a = 1};

// CHECK:STDOUT: --- fail_too_few_values.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.2: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {.a: i32, .b: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32, .b: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_35: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_36: {.a: i32} = struct_literal (%.loc10_35)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
