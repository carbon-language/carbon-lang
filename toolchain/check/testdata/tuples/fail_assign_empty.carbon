// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+3]]:17: ERROR: Cannot initialize tuple of 1 element(s) from tuple with 0 element(s).
// CHECK:STDERR: var x: (i32,) = ();
// CHECK:STDERR:                 ^~
var x: (i32,) = ();

// CHECK:STDOUT: --- fail_assign_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x} [template]
// CHECK:STDOUT:   %.loc10_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc10_13.2: type = converted %.loc10_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref (i32,) = var x
// CHECK:STDOUT:   %x: ref (i32,) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_18: () = tuple_literal ()
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
