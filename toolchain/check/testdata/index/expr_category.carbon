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
// CHECK:STDOUT:   %.loc7_17: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc7_18.1: type = array_type %.loc7_17, i32, const
// CHECK:STDOUT:   %.loc7_18.2: type = ptr_type [i32; 3], const
// CHECK:STDOUT:   %.loc9_15: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc9_16: type = array_type %.loc9_15, i32, const
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32, const
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1, const
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 2, const
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc10_29: type = tuple_type (i32, i32, i32), const
// CHECK:STDOUT:   %.loc13: i32 = int_literal 0, const
// CHECK:STDOUT:   %.loc14_5: i32 = int_literal 0, const
// CHECK:STDOUT:   %.loc14_10: i32 = int_literal 4, const
// CHECK:STDOUT:   %.loc17_26: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc17_27: type = array_type %.loc17_26, i32, const
// CHECK:STDOUT:   %.loc18_16: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc18_17: type = array_type %.loc18_16, i32, const
// CHECK:STDOUT:   %.loc18_22: i32 = int_literal 1, const
// CHECK:STDOUT:   %.loc18_25: i32 = int_literal 2, const
// CHECK:STDOUT:   %.loc18_28: i32 = int_literal 3, const
// CHECK:STDOUT:   %.loc22: i32 = int_literal 0, const
// CHECK:STDOUT:   %.loc23: i32 = int_literal 0, const
// CHECK:STDOUT:   %.loc24: i32 = int_literal 0, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.F = %F, .G = %G, .ValueBinding = %ValueBinding}
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT:   %G: <function> = fn_decl @G, const
// CHECK:STDOUT:   %ValueBinding: <function> = fn_decl @ValueBinding, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: [i32; 3];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 3, const = constants.%.loc10_16
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32, const = constants.%.loc10_17
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1, const = constants.%.loc10_22
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 2, const = constants.%.loc10_25
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 3, const = constants.%.loc10_28
// CHECK:STDOUT:   %.loc10_29.1: (i32, i32, i32) = tuple_literal (%.loc10_22, %.loc10_25, %.loc10_28)
// CHECK:STDOUT:   %.loc10_29.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_29.3: ref i32 = array_index %a.var, %.loc10_29.2
// CHECK:STDOUT:   %.loc10_29.4: init i32 = initialize_from %.loc10_22 to %.loc10_29.3
// CHECK:STDOUT:   %.loc10_29.5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_29.6: ref i32 = array_index %a.var, %.loc10_29.5
// CHECK:STDOUT:   %.loc10_29.7: init i32 = initialize_from %.loc10_25 to %.loc10_29.6
// CHECK:STDOUT:   %.loc10_29.8: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_29.9: ref i32 = array_index %a.var, %.loc10_29.8
// CHECK:STDOUT:   %.loc10_29.10: init i32 = initialize_from %.loc10_28 to %.loc10_29.9
// CHECK:STDOUT:   %.loc10_29.11: init [i32; 3] = array_init (%.loc10_29.4, %.loc10_29.7, %.loc10_29.10) to %a.var
// CHECK:STDOUT:   %.loc10_29.12: init [i32; 3] = converted %.loc10_29.1, %.loc10_29.11
// CHECK:STDOUT:   assign %a.var, %.loc10_29.12
// CHECK:STDOUT:   %.loc13_14: type = ptr_type i32, const
// CHECK:STDOUT:   %pa.var: ref i32* = var pa
// CHECK:STDOUT:   %pa: ref i32* = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc13: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc13_21: i32 = int_literal 0, const = constants.%.loc13
// CHECK:STDOUT:   %.loc13_22: ref i32 = array_index %a.ref.loc13, %.loc13_21
// CHECK:STDOUT:   %.loc13_18: i32* = addr_of %.loc13_22
// CHECK:STDOUT:   assign %pa.var, %.loc13_18
// CHECK:STDOUT:   %a.ref.loc14: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc14_5: i32 = int_literal 0, const = constants.%.loc14_5
// CHECK:STDOUT:   %.loc14_6: ref i32 = array_index %a.ref.loc14, %.loc14_5
// CHECK:STDOUT:   %.loc14_10: i32 = int_literal 4, const = constants.%.loc14_10
// CHECK:STDOUT:   assign %.loc14_6, %.loc14_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_16: i32 = int_literal 3, const = constants.%.loc18_16
// CHECK:STDOUT:   %.loc18_17: type = array_type %.loc18_16, i32, const = constants.%.loc18_17
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc18_22: i32 = int_literal 1, const = constants.%.loc18_22
// CHECK:STDOUT:   %.loc18_25: i32 = int_literal 2, const = constants.%.loc18_25
// CHECK:STDOUT:   %.loc18_28: i32 = int_literal 3, const = constants.%.loc18_28
// CHECK:STDOUT:   %.loc18_29.1: (i32, i32, i32) = tuple_literal (%.loc18_22, %.loc18_25, %.loc18_28)
// CHECK:STDOUT:   %.loc18_29.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc18_29.3: ref i32 = array_index %a.var, %.loc18_29.2
// CHECK:STDOUT:   %.loc18_29.4: init i32 = initialize_from %.loc18_22 to %.loc18_29.3
// CHECK:STDOUT:   %.loc18_29.5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc18_29.6: ref i32 = array_index %a.var, %.loc18_29.5
// CHECK:STDOUT:   %.loc18_29.7: init i32 = initialize_from %.loc18_25 to %.loc18_29.6
// CHECK:STDOUT:   %.loc18_29.8: i32 = int_literal 2
// CHECK:STDOUT:   %.loc18_29.9: ref i32 = array_index %a.var, %.loc18_29.8
// CHECK:STDOUT:   %.loc18_29.10: init i32 = initialize_from %.loc18_28 to %.loc18_29.9
// CHECK:STDOUT:   %.loc18_29.11: init [i32; 3] = array_init (%.loc18_29.4, %.loc18_29.7, %.loc18_29.10) to %a.var
// CHECK:STDOUT:   %.loc18_29.12: init [i32; 3] = converted %.loc18_29.1, %.loc18_29.11
// CHECK:STDOUT:   assign %a.var, %.loc18_29.12
// CHECK:STDOUT:   %a.ref: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc22_5: i32 = int_literal 0, const = constants.%.loc22
// CHECK:STDOUT:   %.loc22_6: ref i32 = array_index %a.ref, %.loc22_5
// CHECK:STDOUT:   %b.ref: [i32; 3] = name_ref b, %b
// CHECK:STDOUT:   %.loc23_5: i32 = int_literal 0, const = constants.%.loc23
// CHECK:STDOUT:   %.loc23_6.1: ref [i32; 3] = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc23_6.2: ref i32 = array_index %.loc23_6.1, %.loc23_5
// CHECK:STDOUT:   %.loc23_6.3: i32 = bind_value %.loc23_6.2
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F, const = file.%F
// CHECK:STDOUT:   %.loc24_4.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc24_4.2: init [i32; 3] = call %F.ref() to %.loc24_4.1
// CHECK:STDOUT:   %.loc24_7: i32 = int_literal 0, const = constants.%.loc24
// CHECK:STDOUT:   %.loc24_4.3: ref [i32; 3] = temporary %.loc24_4.1, %.loc24_4.2
// CHECK:STDOUT:   %.loc24_8.1: ref i32 = array_index %.loc24_4.3, %.loc24_7
// CHECK:STDOUT:   %.loc24_8.2: i32 = bind_value %.loc24_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
