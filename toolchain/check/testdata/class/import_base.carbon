// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

base class Base {
  fn F[self: Self]();
  fn Unused[self: Self]();

  var x: i32;
  var unused: i32;
}

class Child {
  extend base: Base;
}

// --- b.carbon

library "b" api;

import library "a";

fn Run() {
  var a: Child = {.base = {.x = 0, .unused = 1}};
  a.x = 2;
  a.F();
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Child, Base [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: Base} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Child = %Child.decl} [template]
// CHECK:STDOUT:   %Base.decl = class_decl @Base, () [template = constants.%Base]
// CHECK:STDOUT:   %Child.decl = class_decl @Child, () [template = constants.%Child]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Unused: <function> = fn_decl @Unused [template]
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Base> = field_decl unused, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .Unused = %Unused
// CHECK:STDOUT:   .x = %.loc8
// CHECK:STDOUT:   .unused = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: <unbound element of class Child> = base_decl Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unused[%self: Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Child, Base [template]
// CHECK:STDOUT:   %.3: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: {.x: i32, .unused: i32}*} [template]
// CHECK:STDOUT:   %.6: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: {.x: i32, .unused: i32}} [template]
// CHECK:STDOUT:   %.10: Base = struct_value (%.7, %.8) [template]
// CHECK:STDOUT:   %.11: Child = struct_value (%.10) [template]
// CHECK:STDOUT:   %.12: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.14: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %import_ref.1, .Child = %import_ref.2, .Run = %Run} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+19, used [template = constants.%Child]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %import_ref = import_ref ir1, inst+24, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %import_ref
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+6, used [template = imports.%F]
// CHECK:STDOUT:   %import_ref.2: <unbound element of class Base> = import_ref ir1, inst+12, used [template = imports.%.1]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+10, unused
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %import_ref.1
// CHECK:STDOUT:   .x = %import_ref.2
// CHECK:STDOUT:   .Unused = %import_ref.3
// CHECK:STDOUT:   .unused = %import_ref.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Child.decl = class_decl @Child, () [template = constants.%Child]
// CHECK:STDOUT:   %Base.decl = class_decl @Base, () [template = constants.%Base]
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, file.%import_ref.2 [template = constants.%Child]
// CHECK:STDOUT:   %a.var: ref Child = var a
// CHECK:STDOUT:   %a: ref Child = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_33: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_46: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_47.1: {.x: i32, .unused: i32} = struct_literal (%.loc7_33, %.loc7_46)
// CHECK:STDOUT:   %.loc7_48.1: {.base: {.x: i32, .unused: i32}} = struct_literal (%.loc7_47.1)
// CHECK:STDOUT:   %.loc7_48.2: ref Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_47.2: ref i32 = class_element_access %.loc7_48.2, element0
// CHECK:STDOUT:   %.loc7_47.3: init i32 = initialize_from %.loc7_33 to %.loc7_47.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_47.4: ref i32 = class_element_access %.loc7_48.2, element1
// CHECK:STDOUT:   %.loc7_47.5: init i32 = initialize_from %.loc7_46 to %.loc7_47.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_47.6: init Base = class_init (%.loc7_47.3, %.loc7_47.5), %.loc7_48.2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_47.7: init Base = converted %.loc7_47.1, %.loc7_47.6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_48.3: init Child = class_init (%.loc7_47.7), %a.var [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_48.4: init Child = converted %.loc7_48.1, %.loc7_48.3 [template = constants.%.11]
// CHECK:STDOUT:   assign %a.var, %.loc7_48.4
// CHECK:STDOUT:   %a.ref.loc8: ref Child = name_ref a, %a
// CHECK:STDOUT:   %.loc8_4.1: ref Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_3: ref Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.2: ref i32 = class_element_access %.loc8_3, element0
// CHECK:STDOUT:   %.loc8_9: i32 = int_literal 2 [template = constants.%.13]
// CHECK:STDOUT:   assign %.loc8_4.2, %.loc8_9
// CHECK:STDOUT:   %a.ref.loc9: ref Child = name_ref a, %a
// CHECK:STDOUT:   %.loc9_4: <bound method> = bound_method %a.ref.loc9, @Base.%import_ref.1
// CHECK:STDOUT:   %.loc9_6.1: ref Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_3.1: ref Base = converted %a.ref.loc9, %.loc9_6.1
// CHECK:STDOUT:   %.loc9_3.2: Base = bind_value %.loc9_3.1
// CHECK:STDOUT:   %.loc9_6.2: init () = call %.loc9_4(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Base]();
// CHECK:STDOUT:
