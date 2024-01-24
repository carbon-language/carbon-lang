// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 102 [template]
// CHECK:STDOUT:   %.6: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .y = %y} [template]
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_25: i32 = int_literal 102 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_28.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_25)
// CHECK:STDOUT:   %.loc7_28.2: (i32, i32) = tuple_value (%.loc7_22, %.loc7_25) [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_28.3: (i32, i32) = converted %.loc7_28.1, %.loc7_28.2 [template = constants.%.6]
// CHECK:STDOUT:   %v: (i32, i32) = bind_name v, %.loc7_28.3
// CHECK:STDOUT:   %.loc8_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc8_17.2: type = converted %.loc8_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.ref: (i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %w: (i32, i32) = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc10_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_17.2: type = converted %.loc10_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref (i32, i32) = var x
// CHECK:STDOUT:   %x: ref (i32, i32) = bind_name x, %x.var
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 102 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_28.1: (i32, i32) = tuple_literal (%.loc10_22, %.loc10_25)
// CHECK:STDOUT:   %.loc10_28.2: ref i32 = tuple_access %x.var, element0
// CHECK:STDOUT:   %.loc10_28.3: init i32 = initialize_from %.loc10_22 to %.loc10_28.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_28.4: ref i32 = tuple_access %x.var, element1
// CHECK:STDOUT:   %.loc10_28.5: init i32 = initialize_from %.loc10_25 to %.loc10_28.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_28.6: init (i32, i32) = tuple_init (%.loc10_28.3, %.loc10_28.5) to %x.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_28.7: init (i32, i32) = converted %.loc10_28.1, %.loc10_28.6 [template = constants.%.6]
// CHECK:STDOUT:   assign %x.var, %.loc10_28.7
// CHECK:STDOUT:   %.loc11_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc11_17.2: type = converted %.loc11_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %y.var: ref (i32, i32) = var y
// CHECK:STDOUT:   %y: ref (i32, i32) = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref (i32, i32) = name_ref x, %x
// CHECK:STDOUT:   %.loc11_21.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc11_21.2: i32 = bind_value %.loc11_21.1
// CHECK:STDOUT:   %.loc11_21.3: ref i32 = tuple_access %y.var, element0
// CHECK:STDOUT:   %.loc11_21.4: init i32 = initialize_from %.loc11_21.2 to %.loc11_21.3
// CHECK:STDOUT:   %.loc11_21.5: ref i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc11_21.6: i32 = bind_value %.loc11_21.5
// CHECK:STDOUT:   %.loc11_21.7: ref i32 = tuple_access %y.var, element1
// CHECK:STDOUT:   %.loc11_21.8: init i32 = initialize_from %.loc11_21.6 to %.loc11_21.7
// CHECK:STDOUT:   %.loc11_21.9: init (i32, i32) = tuple_init (%.loc11_21.4, %.loc11_21.8) to %y.var
// CHECK:STDOUT:   %.loc11_21.10: init (i32, i32) = converted %x.ref, %.loc11_21.9
// CHECK:STDOUT:   assign %y.var, %.loc11_21.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
