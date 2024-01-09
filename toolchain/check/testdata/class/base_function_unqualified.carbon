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
// CHECK:STDOUT:   %.loc9_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc9_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %.loc16: type = struct_type {.base: Base}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Derived = %Derived.decl}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %.loc7: type = reify_constant %Base, %Base
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived
// CHECK:STDOUT:   %.loc11: type = reify_constant %Derived, %Derived
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc12_20.1: type = unbound_element_type Derived, Base
// CHECK:STDOUT:   %.loc12_20.2: type = reify_constant %.loc12_20.1, %.loc12_20.1
// CHECK:STDOUT:   %.loc12_20.3: <unbound element of class Derived> = base_decl Base, element0
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12_20.3
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F
// CHECK:STDOUT:   %.loc14_12: <function> = reify_constant %F.ref, @Base.%F
// CHECK:STDOUT:   %.loc14_13: init () = call @Base.%F()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F
// CHECK:STDOUT:   %.loc19_3: <function> = reify_constant %F.ref, @Base.%F
// CHECK:STDOUT:   %.loc19_4: init () = call @Base.%F()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
