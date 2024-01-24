// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+3]]:8: ERROR: Cannot implicitly convert from `{.a: i32}` to `type`.
// CHECK:STDERR: var x: {.a = 1};
// CHECK:STDERR:        ^~~~~~~~
var x: {.a = 1};

// CHECK:STDOUT: --- fail_value_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_15: {.a: i32} = struct_literal (%.loc10_14)
// CHECK:STDOUT:   %x.var: ref <error> = var x
// CHECK:STDOUT:   %x: ref <error> = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
