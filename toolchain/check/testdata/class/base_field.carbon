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

fn Access(p: Derived*) -> i32* {
  return &(*p).c;
}

// CHECK:STDOUT: --- base_field.carbon
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
// CHECK:STDOUT:   %.7: type = ptr_type Derived [template]
// CHECK:STDOUT:   %.8: type = ptr_type i32 [template]
// CHECK:STDOUT:   %Access: type = fn_type @Access [template]
// CHECK:STDOUT:   %.9: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Access = struct_value () [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.base: Base, .d: i32, .e: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Access.decl: Access = fn_decl @Access [template = constants.%struct] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc20_21: type = ptr_type Derived [template = constants.%.7]
// CHECK:STDOUT:     %p.loc20_11.1: Derived* = param p
// CHECK:STDOUT:     @Access.%p: Derived* = bind_name p, %p.loc20_11.1
// CHECK:STDOUT:     %.loc20_30: type = ptr_type i32 [template = constants.%.8]
// CHECK:STDOUT:     @Access.%return: ref i32* = var <return slot>
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
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p: Derived*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc21_12: ref Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: <unbound element of class Base> = name_ref c, @Base.%.loc10 [template = @Base.%.loc10]
// CHECK:STDOUT:   %.loc21_15.1: ref Base = class_element_access %.loc21_12, element0
// CHECK:STDOUT:   %.loc21_15.2: ref Base = converted %.loc21_12, %.loc21_15.1
// CHECK:STDOUT:   %.loc21_15.3: ref i32 = class_element_access %.loc21_15.2, element2
// CHECK:STDOUT:   %.loc21_10: i32* = addr_of %.loc21_15.3
// CHECK:STDOUT:   return %.loc21_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
