// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var a: i32;

  fn F[addr self: Self*]();
}

fn Base.F[addr self: Base*]() {
  (*self).a = 1;
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  (*p).F();
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type Base [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.9: type = ptr_type Derived [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.2: Call = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: {.a: i32}*} [template]
// CHECK:STDOUT:   %.11: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc13_26: type = ptr_type Base [template = constants.%.2]
// CHECK:STDOUT:     %self.loc13_16.1: Base* = param self
// CHECK:STDOUT:     @F.%self: Base* = bind_name self, %self.loc13_16.1
// CHECK:STDOUT:     @F.%.loc13: Base* = addr_pattern @F.%self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.2] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc21: type = ptr_type Derived [template = constants.%.9]
// CHECK:STDOUT:     %p.loc21_9.1: Derived* = param p
// CHECK:STDOUT:     @Call.%p: Derived* = bind_name p, %p.loc21_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc10_23: type = ptr_type Base [template = constants.%.2]
// CHECK:STDOUT:     %self.loc10_13.1: Base* = param self
// CHECK:STDOUT:     %self.loc10_13.3: Base* = bind_name self, %self.loc10_13.1
// CHECK:STDOUT:     %.loc10_8: Base* = addr_pattern %self.loc10_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc18: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self: Base*]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base* = name_ref self, %self
// CHECK:STDOUT:   %.loc14_4: ref Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: <unbound element of class Base> = name_ref a, @Base.%.loc8 [template = @Base.%.loc8]
// CHECK:STDOUT:   %.loc14_10: ref i32 = class_element_access %.loc14_4, element0
// CHECK:STDOUT:   %.loc14_15: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   assign %.loc14_10, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: Derived*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc22_4.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Base.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc22_7: <bound method> = bound_method %.loc22_4.1, %F.ref
// CHECK:STDOUT:   %.loc22_4.2: Derived* = addr_of %.loc22_4.1
// CHECK:STDOUT:   %.loc22_9.1: ref Derived = deref %.loc22_4.2
// CHECK:STDOUT:   %.loc22_9.2: ref Base = class_element_access %.loc22_9.1, element0
// CHECK:STDOUT:   %.loc22_9.3: Base* = addr_of %.loc22_9.2
// CHECK:STDOUT:   %.loc22_9.4: Base* = converted %.loc22_4.2, %.loc22_9.3
// CHECK:STDOUT:   %F.call: init () = call %.loc22_7(%.loc22_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
