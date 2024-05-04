// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:13: ERROR: Cannot initialize tuple of 0 element(s) from tuple with 1 element(s).
// CHECK:STDERR: var x: () = ((),);
// CHECK:STDERR:             ^~~~~
var x: () = ((),);

// CHECK:STDOUT: --- fail_assign_to_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (()) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_15: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_17: ((),) = tuple_literal (%.loc10_15)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
