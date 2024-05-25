// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.v: ()} = {.v = ()};

// CHECK:STDOUT: --- global_decl_with_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: ()} [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %struct: {.v: ()} = struct_value (%tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_14.2: type = converted %.loc7_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = struct_type {.v: ()} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref {.v: ()} = var x
// CHECK:STDOUT:   %x: ref {.v: ()} = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_26.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_27.1: {.v: ()} = struct_literal (%.loc7_26.1)
// CHECK:STDOUT:   %.loc7_27.2: ref () = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc7_26.2: init () = tuple_init () to %.loc7_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.3: init () = converted %.loc7_26.1, %.loc7_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_27.4: init {.v: ()} = struct_init (%.loc7_27.3) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_28: init {.v: ()} = converted %.loc7_27.1, %.loc7_27.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_28
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
