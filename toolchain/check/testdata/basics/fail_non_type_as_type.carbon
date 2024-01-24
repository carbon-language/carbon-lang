// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_non_type_as_type.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var x: type = 42;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var x: type = 42;

// CHECK:STDOUT: --- fail_non_type_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %x.var: ref type = var x
// CHECK:STDOUT:   %x: ref type = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10: i32 = int_literal 42 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
