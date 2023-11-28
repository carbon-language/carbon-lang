// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `i32` to `() as type`.
// CHECK:STDERR: var x: () = (66);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var x: () = (66);

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_assign_to_empty.carbon" {
// CHECK:STDOUT:   %.loc10_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.loc10
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 66
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
