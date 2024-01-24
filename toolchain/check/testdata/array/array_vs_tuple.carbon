// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G() {
  // These should have two different constant values.
  var a: [i32; 3] = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.10: [i32; 3] = tuple_value (%.4, %.5, %.1) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.12: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.13: (i32, i32, i32) = tuple_value (%.4, %.5, %.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.G = %G} [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_17: type = array_type %.loc9_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_25: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_28: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_29.1: (i32, i32, i32) = tuple_literal (%.loc9_22, %.loc9_25, %.loc9_28)
// CHECK:STDOUT:   %.loc9_29.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_29.3: ref i32 = array_index %a.var, %.loc9_29.2
// CHECK:STDOUT:   %.loc9_29.4: init i32 = initialize_from %.loc9_22 to %.loc9_29.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_29.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_29.6: ref i32 = array_index %a.var, %.loc9_29.5
// CHECK:STDOUT:   %.loc9_29.7: init i32 = initialize_from %.loc9_25 to %.loc9_29.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_29.8: i32 = int_literal 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_29.9: ref i32 = array_index %a.var, %.loc9_29.8
// CHECK:STDOUT:   %.loc9_29.10: init i32 = initialize_from %.loc9_28 to %.loc9_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_29.11: init [i32; 3] = array_init (%.loc9_29.4, %.loc9_29.7, %.loc9_29.10) to %a.var [template = constants.%.10]
// CHECK:STDOUT:   %.loc9_29.12: init [i32; 3] = converted %.loc9_29.1, %.loc9_29.11 [template = constants.%.10]
// CHECK:STDOUT:   assign %a.var, %.loc9_29.12
// CHECK:STDOUT:   %.loc10_24.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:   %b.var: ref (i32, i32, i32) = var b
// CHECK:STDOUT:   %b: ref (i32, i32, i32) = bind_name b, %b.var
// CHECK:STDOUT:   %.loc10_29: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_32: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_35: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_36.1: (i32, i32, i32) = tuple_literal (%.loc10_29, %.loc10_32, %.loc10_35)
// CHECK:STDOUT:   %.loc10_36.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc10_36.3: init i32 = initialize_from %.loc10_29 to %.loc10_36.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_36.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc10_36.5: init i32 = initialize_from %.loc10_32 to %.loc10_36.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_36.6: ref i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc10_36.7: init i32 = initialize_from %.loc10_35 to %.loc10_36.6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_36.8: init (i32, i32, i32) = tuple_init (%.loc10_36.3, %.loc10_36.5, %.loc10_36.7) to %b.var [template = constants.%.13]
// CHECK:STDOUT:   %.loc10_36.9: init (i32, i32, i32) = converted %.loc10_36.1, %.loc10_36.8 [template = constants.%.13]
// CHECK:STDOUT:   assign %b.var, %.loc10_36.9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
