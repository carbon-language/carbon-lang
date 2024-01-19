// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+3]]:8: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var a: [1; 39999999999999999993];
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~
var a: [1; 39999999999999999993];

// CHECK:STDOUT: --- fail_bound_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 39999999999999999993 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, <error> [template]
// CHECK:STDOUT:   %.4: type = ptr_type [<error>; 3106511852580896761] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.a = %a} [template]
// CHECK:STDOUT:   %.loc10_9: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 39999999999999999993 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_32: type = array_type %.loc10_12, <error> [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref [<error>; 3106511852580896761] = var a
// CHECK:STDOUT:   %a: ref [<error>; 3106511852580896761] = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
