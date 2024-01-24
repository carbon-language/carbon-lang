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
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.8: type = ptr_type Derived [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: {.a: i32}*} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.11: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Derived = %Derived.decl, .Call = %Call} [template]
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:   %.loc18: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self: Base*]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base* = name_ref self, %self
// CHECK:STDOUT:   %.loc14_4: ref Base = deref %self.ref
// CHECK:STDOUT:   %.loc14_10: ref i32 = class_element_access %.loc14_4, element0
// CHECK:STDOUT:   %.loc14_15: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   assign %.loc14_10, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: Derived*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc22_4.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc22_7: <bound method> = bound_method %.loc22_4.1, @Base.%F
// CHECK:STDOUT:   %.loc22_4.2: Derived* = addr_of %.loc22_4.1
// CHECK:STDOUT:   %.loc22_9.1: ref Derived = deref %.loc22_4.2
// CHECK:STDOUT:   %.loc22_9.2: ref Base = class_element_access %.loc22_9.1, element0
// CHECK:STDOUT:   %.loc22_9.3: Base* = addr_of %.loc22_9.2
// CHECK:STDOUT:   %.loc22_4.3: Base* = converted %.loc22_4.2, %.loc22_9.3
// CHECK:STDOUT:   %.loc22_9.4: init () = call %.loc22_7(%.loc22_4.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
