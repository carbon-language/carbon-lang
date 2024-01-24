// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:13: ERROR: Cannot initialize struct with 0 field(s) from struct with 1 field(s).
// CHECK:STDERR: var x: {} = {.a = 1};
// CHECK:STDERR:             ^~~~~~~~
var x: {} = {.a = 1};

// CHECK:STDOUT: --- fail_assign_to_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {} = var x
// CHECK:STDOUT:   %x: ref {} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_20: {.a: i32} = struct_literal (%.loc10_19)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
