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
// CHECK:STDOUT:   %.loc11_1.1: type = struct_type {.a: i32, .b: i32, .c: i32}, const
// CHECK:STDOUT:   %.loc11_1.2: type = ptr_type {.a: i32, .b: i32, .c: i32}, const
// CHECK:STDOUT:   %.loc18_1.1: type = struct_type {.base: Base, .d: i32, .e: i32}, const
// CHECK:STDOUT:   %.loc18_1.2: type = struct_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32}, const
// CHECK:STDOUT:   %.loc18_1.3: type = ptr_type {.base: {.a: i32, .b: i32, .c: i32}*, .d: i32, .e: i32}, const
// CHECK:STDOUT:   %.loc13: type = ptr_type {.base: Base, .d: i32, .e: i32}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Base = %Base.decl, .Derived = %Derived.decl, .Access = %Access}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base, const
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived, const
// CHECK:STDOUT:   %Access: <function> = fn_decl @Access, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Base, i32, const
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class Base> = field_decl a, element0, const
// CHECK:STDOUT:   %a: <unbound element of class Base> = bind_name a, %.loc8_8.2, const = %.loc8_8.2
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_element_type Base, i32, const
// CHECK:STDOUT:   %.loc9_8.2: <unbound element of class Base> = field_decl b, element1, const
// CHECK:STDOUT:   %b: <unbound element of class Base> = bind_name b, %.loc9_8.2, const = %.loc9_8.2
// CHECK:STDOUT:   %.loc10_8.1: type = unbound_element_type Base, i32, const
// CHECK:STDOUT:   %.loc10_8.2: <unbound element of class Base> = field_decl c, element2, const
// CHECK:STDOUT:   %c: <unbound element of class Base> = bind_name c, %.loc10_8.2, const = %.loc10_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT:   .c = %c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base, const = file.%Base
// CHECK:STDOUT:   %.loc14_20.1: type = unbound_element_type Derived, Base, const
// CHECK:STDOUT:   %.loc14_20.2: <unbound element of class Derived> = base_decl Base, element0, const
// CHECK:STDOUT:   %.loc16_8.1: type = unbound_element_type Derived, i32, const
// CHECK:STDOUT:   %.loc16_8.2: <unbound element of class Derived> = field_decl d, element1, const
// CHECK:STDOUT:   %d: <unbound element of class Derived> = bind_name d, %.loc16_8.2, const = %.loc16_8.2
// CHECK:STDOUT:   %.loc17_8.1: type = unbound_element_type Derived, i32, const
// CHECK:STDOUT:   %.loc17_8.2: <unbound element of class Derived> = field_decl e, element2, const
// CHECK:STDOUT:   %e: <unbound element of class Derived> = bind_name e, %.loc17_8.2, const = %.loc17_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc14_20.2
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT:   .e = %e
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p: Derived*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc21_12: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc21_15.1: ref Base = class_element_access %.loc21_12, element0
// CHECK:STDOUT:   %.loc21_15.2: ref Base = converted %.loc21_12, %.loc21_15.1
// CHECK:STDOUT:   %.loc21_15.3: ref i32 = class_element_access %.loc21_15.2, element2
// CHECK:STDOUT:   %.loc21_10: i32* = addr_of %.loc21_15.3
// CHECK:STDOUT:   return %.loc21_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
