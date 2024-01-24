// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;
  var b: i32;
}

fn F() -> C {
  returned var result: C = {.a = 1, .b = 2};
  return var;
}

fn G() -> i32 {
  returned var result: i32 = 0;
  return var;
}

// CHECK:STDOUT: --- returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type C, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: C = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.C = %C.decl, .F = %F, .G = %G} [template]
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8: <unbound element of class C> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class C> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %result: ref C = bind_name result, %return
// CHECK:STDOUT:   %.loc13_34: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_42: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_43.1: {.a: i32, .b: i32} = struct_literal (%.loc13_34, %.loc13_42)
// CHECK:STDOUT:   %.loc13_43.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc13_43.3: init i32 = initialize_from %.loc13_34 to %.loc13_43.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_43.4: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc13_43.5: init i32 = initialize_from %.loc13_42 to %.loc13_43.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_43.6: init C = class_init (%.loc13_43.3, %.loc13_43.5), %return [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_43.7: init C = converted %.loc13_43.1, %.loc13_43.6 [template = constants.%.6]
// CHECK:STDOUT:   assign %return, %.loc13_43.7
// CHECK:STDOUT:   return %result
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %result.var: ref i32 = var result
// CHECK:STDOUT:   %result: ref i32 = bind_name result, %result.var
// CHECK:STDOUT:   %.loc18_30: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   assign %result.var, %.loc18_30
// CHECK:STDOUT:   %.loc18_16: i32 = bind_value %result
// CHECK:STDOUT:   return %.loc18_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
