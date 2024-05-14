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
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %Unused: type = fn_type @Unused [template]
// CHECK:STDOUT:   %struct.2: Unused = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Child, Base [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: Base} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Child = %Child.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Child.decl: type = class_decl @Child [template = constants.%Child] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc5: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.loc5_8.1: Base = param self
// CHECK:STDOUT:     %self.loc5_8.2: Base = bind_name self, %self.loc5_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unused.decl: Unused = fn_decl @Unused [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc6: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.loc6_13.1: Base = param self
// CHECK:STDOUT:     %self.loc6_13.2: Base = bind_name self, %self.loc6_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Base> = field_decl unused, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Unused = %Unused.decl
// CHECK:STDOUT:   .x = %.loc8
// CHECK:STDOUT:   .unused = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc13: <unbound element of class Child> = base_decl Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Child
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Base.%self.loc5_8.2: Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unused[@Base.%self.loc6_13.2: Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Run = struct_value () [template]
// CHECK:STDOUT:   %Child: type = class_type @Child [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.2: type = struct_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type Child, Base [template]
// CHECK:STDOUT:   %.4: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.x: i32, .unused: i32} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: {.x: i32, .unused: i32}*} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: {.x: i32, .unused: i32}} [template]
// CHECK:STDOUT:   %struct.2: Base = struct_value (%.8, %.9) [template]
// CHECK:STDOUT:   %struct.3: Child = struct_value (%struct.2) [template]
// CHECK:STDOUT:   %.11: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.4: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = %import_ref.1
// CHECK:STDOUT:     .Child = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+25, loaded [template = constants.%Child]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref.3: F = import_ref ir1, inst+7, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.5: <unbound element of class Base> = import_ref ir1, inst+18, loaded [template = imports.%.1]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+30, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Child {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.8
// CHECK:STDOUT:   .base = file.%import_ref.9
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = file.%import_ref.3
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   .x = file.%import_ref.5
// CHECK:STDOUT:   .Unused = file.%import_ref.6
// CHECK:STDOUT:   .unused = file.%import_ref.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Child.ref: type = name_ref Child, file.%import_ref.2 [template = constants.%Child]
// CHECK:STDOUT:   %a.var: ref Child = var a
// CHECK:STDOUT:   %a: ref Child = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_33: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_46: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_47.1: {.x: i32, .unused: i32} = struct_literal (%.loc7_33, %.loc7_46)
// CHECK:STDOUT:   %.loc7_48.1: {.base: {.x: i32, .unused: i32}} = struct_literal (%.loc7_47.1)
// CHECK:STDOUT:   %.loc7_48.2: ref Base = class_element_access %a.var, element0
// CHECK:STDOUT:   %.loc7_47.2: ref i32 = class_element_access %.loc7_48.2, element0
// CHECK:STDOUT:   %.loc7_47.3: init i32 = initialize_from %.loc7_33 to %.loc7_47.2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_47.4: ref i32 = class_element_access %.loc7_48.2, element1
// CHECK:STDOUT:   %.loc7_47.5: init i32 = initialize_from %.loc7_46 to %.loc7_47.4 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_47.6: init Base = class_init (%.loc7_47.3, %.loc7_47.5), %.loc7_48.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_48.3: init Base = converted %.loc7_47.1, %.loc7_47.6 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7_48.4: init Child = class_init (%.loc7_48.3), %a.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc7_49: init Child = converted %.loc7_48.1, %.loc7_48.4 [template = constants.%struct.3]
// CHECK:STDOUT:   assign %a.var, %.loc7_49
// CHECK:STDOUT:   %a.ref.loc8: ref Child = name_ref a, %a
// CHECK:STDOUT:   %x.ref: <unbound element of class Base> = name_ref x, file.%import_ref.5 [template = imports.%.1]
// CHECK:STDOUT:   %.loc8_4.1: ref Base = class_element_access %a.ref.loc8, element0
// CHECK:STDOUT:   %.loc8_4.2: ref Base = converted %a.ref.loc8, %.loc8_4.1
// CHECK:STDOUT:   %.loc8_4.3: ref i32 = class_element_access %.loc8_4.2, element0
// CHECK:STDOUT:   %.loc8_9: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   assign %.loc8_4.3, %.loc8_9
// CHECK:STDOUT:   %a.ref.loc9: ref Child = name_ref a, %a
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%import_ref.3 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc9_4: <bound method> = bound_method %a.ref.loc9, %F.ref
// CHECK:STDOUT:   %.loc9_6.1: ref Base = class_element_access %a.ref.loc9, element0
// CHECK:STDOUT:   %.loc9_6.2: ref Base = converted %a.ref.loc9, %.loc9_6.1
// CHECK:STDOUT:   %.loc9_6.3: Base = bind_value %.loc9_6.2
// CHECK:STDOUT:   %F.call: init () = call %.loc9_4(%.loc9_6.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Base]();
// CHECK:STDOUT:
