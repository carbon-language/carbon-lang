// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var pa: i32* = &a[0];
  a[0] = 4;
}

fn ValueBinding(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Index but don't do anything else so we can check that a value binding is
  // produced when appropriate.
  a[0];
  b[0];
  F()[0];
}

// CHECK:STDOUT: --- expr_category.carbon
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
// CHECK:STDOUT:   %.11: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .G = %G, .ValueBinding = %ValueBinding} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %ValueBinding: <function> = fn_decl @ValueBinding [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: [i32; 3];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_29.1: (i32, i32, i32) = tuple_literal (%.loc10_22, %.loc10_25, %.loc10_28)
// CHECK:STDOUT:   %.loc10_29.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_29.3: ref i32 = array_index %a.var, %.loc10_29.2
// CHECK:STDOUT:   %.loc10_29.4: init i32 = initialize_from %.loc10_22 to %.loc10_29.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_29.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_29.6: ref i32 = array_index %a.var, %.loc10_29.5
// CHECK:STDOUT:   %.loc10_29.7: init i32 = initialize_from %.loc10_25 to %.loc10_29.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_29.8: i32 = int_literal 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_29.9: ref i32 = array_index %a.var, %.loc10_29.8
// CHECK:STDOUT:   %.loc10_29.10: init i32 = initialize_from %.loc10_28 to %.loc10_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_29.11: init [i32; 3] = array_init (%.loc10_29.4, %.loc10_29.7, %.loc10_29.10) to %a.var [template = constants.%.10]
// CHECK:STDOUT:   %.loc10_29.12: init [i32; 3] = converted %.loc10_29.1, %.loc10_29.11 [template = constants.%.10]
// CHECK:STDOUT:   assign %a.var, %.loc10_29.12
// CHECK:STDOUT:   %.loc13_14: type = ptr_type i32 [template = constants.%.11]
// CHECK:STDOUT:   %pa.var: ref i32* = var pa
// CHECK:STDOUT:   %pa: ref i32* = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc13: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc13_21: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc13_22: ref i32 = array_index %a.ref.loc13, %.loc13_21
// CHECK:STDOUT:   %.loc13_18: i32* = addr_of %.loc13_22
// CHECK:STDOUT:   assign %pa.var, %.loc13_18
// CHECK:STDOUT:   %a.ref.loc14: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc14_5: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc14_6: ref i32 = array_index %a.ref.loc14, %.loc14_5
// CHECK:STDOUT:   %.loc14_10: i32 = int_literal 4 [template = constants.%.13]
// CHECK:STDOUT:   assign %.loc14_6, %.loc14_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_16: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_17: type = array_type %.loc18_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc18_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_25: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_28: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_29.1: (i32, i32, i32) = tuple_literal (%.loc18_22, %.loc18_25, %.loc18_28)
// CHECK:STDOUT:   %.loc18_29.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_29.3: ref i32 = array_index %a.var, %.loc18_29.2
// CHECK:STDOUT:   %.loc18_29.4: init i32 = initialize_from %.loc18_22 to %.loc18_29.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_29.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc18_29.6: ref i32 = array_index %a.var, %.loc18_29.5
// CHECK:STDOUT:   %.loc18_29.7: init i32 = initialize_from %.loc18_25 to %.loc18_29.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_29.8: i32 = int_literal 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_29.9: ref i32 = array_index %a.var, %.loc18_29.8
// CHECK:STDOUT:   %.loc18_29.10: init i32 = initialize_from %.loc18_28 to %.loc18_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_29.11: init [i32; 3] = array_init (%.loc18_29.4, %.loc18_29.7, %.loc18_29.10) to %a.var [template = constants.%.10]
// CHECK:STDOUT:   %.loc18_29.12: init [i32; 3] = converted %.loc18_29.1, %.loc18_29.11 [template = constants.%.10]
// CHECK:STDOUT:   assign %a.var, %.loc18_29.12
// CHECK:STDOUT:   %a.ref: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc22_5: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc22_6: ref i32 = array_index %a.ref, %.loc22_5
// CHECK:STDOUT:   %b.ref: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc23_5: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc23_6.1: ref [i32; 3] = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc23_6.2: ref i32 = array_index %.loc23_6.1, %.loc23_5
// CHECK:STDOUT:   %.loc23_6.3: i32 = bind_value %.loc23_6.2
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc24_4.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc24_4.2: init [i32; 3] = call %F.ref() to %.loc24_4.1
// CHECK:STDOUT:   %.loc24_7: i32 = int_literal 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc24_4.3: ref [i32; 3] = temporary %.loc24_4.1, %.loc24_4.2
// CHECK:STDOUT:   %.loc24_8.1: ref i32 = array_index %.loc24_4.3, %.loc24_7
// CHECK:STDOUT:   %.loc24_8.2: i32 = bind_value %.loc24_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
