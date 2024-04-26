// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {} = {};
var y: {} = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {} = var x
// CHECK:STDOUT:   %x: ref {} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc8_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc8_9.2: type = converted %.loc8_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref {} = var y
// CHECK:STDOUT:   %y: ref {} = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init {} = struct_init () to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_15: init {} = converted %.loc7_14.1, %.loc7_14.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_15
// CHECK:STDOUT:   %x.ref: ref {} = name_ref x, file.%x
// CHECK:STDOUT:   %.loc8_13: init {} = struct_init () to file.%y.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc8_14: init {} = converted %x.ref, %.loc8_13 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%y.var, %.loc8_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
