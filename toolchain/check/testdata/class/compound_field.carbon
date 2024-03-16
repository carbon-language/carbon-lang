// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn AccessDerived(d: Derived) -> i32 {
  return d.(Derived.d);
}

fn AccessBase(d: Derived) -> i32 {
  return d.(Base.b);
}

// CHECK:STDOUT: --- compound_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Derived, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: Base, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.base: Base, .d: i32, .e: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived
// CHECK:STDOUT:     .AccessBase = %AccessBase
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %AccessDerived: <function> = fn_decl @AccessDerived [template] {
// CHECK:STDOUT:     %Derived.ref.loc20: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc20_18.1: Derived = param d
// CHECK:STDOUT:     @AccessDerived.%d: Derived = bind_name d, %d.loc20_18.1
// CHECK:STDOUT:     %return.var.loc20: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase: <function> = fn_decl @AccessBase [template] {
// CHECK:STDOUT:     %Derived.ref.loc24: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.loc24_15.1: Derived = param d
// CHECK:STDOUT:     @AccessBase.%d: Derived = bind_name d, %d.loc24_15.1
// CHECK:STDOUT:     %return.var.loc24: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc9: <unbound element of class Base> = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc10: <unbound element of class Base> = field_decl c, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT:   .b = %.loc9
// CHECK:STDOUT:   .c = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc14: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %.loc16: <unbound element of class Derived> = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc17: <unbound element of class Derived> = field_decl e, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .d = %.loc16
// CHECK:STDOUT:   .e = %.loc17
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d: Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc21_10: Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc21_20: <unbound element of class Derived> = name_ref d, @Derived.%.loc16 [template = @Derived.%.loc16]
// CHECK:STDOUT:   %.loc21_11.1: ref i32 = class_element_access %d.ref.loc21_10, element1
// CHECK:STDOUT:   %.loc21_11.2: i32 = bind_value %.loc21_11.1
// CHECK:STDOUT:   return %.loc21_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d: Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b.ref: <unbound element of class Base> = name_ref b, @Base.%.loc9 [template = @Base.%.loc9]
// CHECK:STDOUT:   %.loc25_11.1: ref Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc25_10: ref Base = converted %d.ref, %.loc25_11.1
// CHECK:STDOUT:   %.loc25_11.2: ref i32 = class_element_access %.loc25_10, element1
// CHECK:STDOUT:   %.loc25_11.3: i32 = bind_value %.loc25_11.2
// CHECK:STDOUT:   return %.loc25_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
