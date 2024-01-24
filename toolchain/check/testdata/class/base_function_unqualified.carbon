// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  fn F();
}

class Derived {
  extend base: Base;

  fn G() { F(); }
  fn H();
}

fn Derived.H() {
  F();
}

// CHECK:STDOUT: --- base_function_unqualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: Base} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Derived = %Derived.decl} [template]
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:   %.loc12: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F [template = @Base.%F]
// CHECK:STDOUT:   %.loc14: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F [template = @Base.%F]
// CHECK:STDOUT:   %.loc19: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
