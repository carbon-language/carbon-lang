// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_invalid_type.carbon:[[@LINE+3]]:9: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var a: [1; 1];
// CHECK:STDERR:         ^
var a: [1; 1];

// CHECK:STDOUT: --- fail_invalid_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a} [template]
// CHECK:STDOUT:   %.loc10_9: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_13: type = array_type %.loc10_12, <error> [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
