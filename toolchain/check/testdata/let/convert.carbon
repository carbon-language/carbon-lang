// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w[1];
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.7: (i32, i32, i32) = tuple_value (%.4, %.5, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_24.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc8_24.2: type = converted %.loc8_24.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.var: ref (i32, i32, i32) = var v
// CHECK:STDOUT:   %v: ref (i32, i32, i32) = bind_name v, %v.var
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_35: i32 = int_literal 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_36.1: (i32, i32, i32) = tuple_literal (%.loc8_29, %.loc8_32, %.loc8_35)
// CHECK:STDOUT:   %.loc8_36.2: ref i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc8_36.3: init i32 = initialize_from %.loc8_29 to %.loc8_36.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_36.4: ref i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc8_36.5: init i32 = initialize_from %.loc8_32 to %.loc8_36.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_36.6: ref i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc8_36.7: init i32 = initialize_from %.loc8_35 to %.loc8_36.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_36.8: init (i32, i32, i32) = tuple_init (%.loc8_36.3, %.loc8_36.5, %.loc8_36.7) to %v.var [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_36.9: init (i32, i32, i32) = converted %.loc8_36.1, %.loc8_36.8 [template = constants.%.7]
// CHECK:STDOUT:   assign %v.var, %.loc8_36.9
// CHECK:STDOUT:   %.loc10_24.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.ref: ref (i32, i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %.loc10_28.1: ref i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc10_28.2: i32 = bind_value %.loc10_28.1
// CHECK:STDOUT:   %.loc10_28.3: ref i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc10_28.4: i32 = bind_value %.loc10_28.3
// CHECK:STDOUT:   %.loc10_28.5: ref i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc10_28.6: i32 = bind_value %.loc10_28.5
// CHECK:STDOUT:   %.loc10_28.7: (i32, i32, i32) = tuple_value (%.loc10_28.2, %.loc10_28.4, %.loc10_28.6)
// CHECK:STDOUT:   %.loc10_28.8: (i32, i32, i32) = converted %v.ref, %.loc10_28.7
// CHECK:STDOUT:   %w: (i32, i32, i32) = bind_name w, %.loc10_28.8
// CHECK:STDOUT:   %w.ref: (i32, i32, i32) = name_ref w, %w
// CHECK:STDOUT:   %.loc11_12: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_13: i32 = tuple_index %w.ref, %.loc11_12
// CHECK:STDOUT:   return %.loc11_13
// CHECK:STDOUT: }
// CHECK:STDOUT:
