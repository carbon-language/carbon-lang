// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32,) = (4,);
var y: (i32,) = x;

// CHECK:STDOUT: --- one_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .y = %y} [template]
// CHECK:STDOUT:   %.loc7_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.2: type = converted %.loc7_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref (i32,) = var x
// CHECK:STDOUT:   %x: ref (i32,) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_20.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_20.2: init (i32,) = tuple_init (%.loc7_18) to %x.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_20.3: init (i32,) = converted %.loc7_20.1, %.loc7_20.2 [template = constants.%.4]
// CHECK:STDOUT:   assign %x.var, %.loc7_20.3
// CHECK:STDOUT:   %.loc8_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc8_13.2: type = converted %.loc8_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %y.var: ref (i32,) = var y
// CHECK:STDOUT:   %y: ref (i32,) = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref (i32,) = name_ref x, %x
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init (%.loc8_17.2) to %y.var
// CHECK:STDOUT:   %.loc8_17.4: init (i32,) = converted %x.ref, %.loc8_17.3
// CHECK:STDOUT:   assign %y.var, %.loc8_17.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
