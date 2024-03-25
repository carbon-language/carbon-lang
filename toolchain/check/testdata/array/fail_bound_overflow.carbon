// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Once we preserve the full value of integer literals in SemIR, check
// that we reject the array bound being too large.

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 39999999999999999993 does not fit in i32.
// CHECK:STDERR: var a: [i32; 39999999999999999993];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: [i32; 39999999999999999993];

// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+7]]:9: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var b: [1; 39999999999999999993];
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_bound_overflow.carbon:[[@LINE+3]]:12: ERROR: Integer literal with value 39999999999999999993 does not fit in i32.
// CHECK:STDERR: var b: [1; 39999999999999999993];
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~
var b: [1; 39999999999999999993];

// CHECK:STDOUT: --- fail_bound_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: type = array_type <error>, i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %.loc23_9: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc23_32: type = array_type <error>, <error> [template = <error>]
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   %b: ref <error> = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
