// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: () = ();
var y: () = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .y = %y} [template]
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init () = tuple_init () to %x.var [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_14.3: init () = converted %.loc7_14.1, %.loc7_14.2 [template = constants.%.2]
// CHECK:STDOUT:   assign %x.var, %.loc7_14.3
// CHECK:STDOUT:   %.loc8_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref () = var y
// CHECK:STDOUT:   %y: ref () = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref () = name_ref x, %x
// CHECK:STDOUT:   %.loc8_13.1: init () = tuple_init () to %y.var [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_13.2: init () = converted %x.ref, %.loc8_13.1 [template = constants.%.2]
// CHECK:STDOUT:   assign %y.var, %.loc8_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
