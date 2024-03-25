// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Once we supoprt negating integer literals, check that we reject a
// negative array bound.

// CHECK:STDERR: fail_bound_negative.carbon:[[@LINE+3]]:14: ERROR: Semantics TODO: `missing or invalid operator interface`.
// CHECK:STDERR: var a: [i32; -1];
// CHECK:STDERR:              ^~
var a: [i32; -1];

// CHECK:STDOUT: --- fail_bound_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_15: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_16: type = array_type <error>, i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
