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
// CHECK:STDOUT:   %.loc9_1.1: type = struct_type {}, const
// CHECK:STDOUT:   %.loc9_1.2: type = tuple_type (), const
// CHECK:STDOUT:   %.loc7: type = ptr_type {}, const
// CHECK:STDOUT:   %.loc16: type = struct_type {.base: Base}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Base = %Base.decl, .Derived = %Derived.decl}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base, const
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived, const
// CHECK:STDOUT:   %H: <function> = fn_decl @H, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base, const = file.%Base
// CHECK:STDOUT:   %.loc12_20.1: type = unbound_element_type Derived, Base, const
// CHECK:STDOUT:   %.loc12_20.2: <unbound element of class Derived> = base_decl Base, element0, const
// CHECK:STDOUT:   %G: <function> = fn_decl @G, const
// CHECK:STDOUT:   %H: <function> = fn_decl @H, const
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12_20.2
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F, const = @Base.%F
// CHECK:STDOUT:   %.loc14: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F, const = @Base.%F
// CHECK:STDOUT:   %.loc19: init () = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
