// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var a: i32;
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> i32;
  fn AddrSelfBase[addr self: Base*]();
}

fn Derived.SelfBase[self: Base]() -> i32 {
  return self.a;
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  (*self).a = 1;
}

fn Call(p: Derived*) -> i32 {
  (*p).AddrSelfBase();
  return (*p).SelfBase();
}

// CHECK:STDOUT: --- self_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc16_1.1: type = struct_type {.base: Base}
// CHECK:STDOUT:   %.loc16_1.2: type = struct_type {.base: {.a: i32}*}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: Base}
// CHECK:STDOUT:   %.loc27: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Derived = %Derived.decl, .Call = %Call}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %.loc7: type = reify_constant %Base, %Base
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived
// CHECK:STDOUT:   %.loc11: type = reify_constant %Derived, %Derived
// CHECK:STDOUT:   %SelfBase: <function> = fn_decl @SelfBase
// CHECK:STDOUT:   %AddrSelfBase: <function> = fn_decl @AddrSelfBase
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Base, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class Base> = field_decl a, element0
// CHECK:STDOUT:   %a: <unbound element of class Base> = bind_name a, %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc12_20.1: type = unbound_element_type Derived, Base
// CHECK:STDOUT:   %.loc12_20.2: type = reify_constant %.loc12_20.1, %.loc12_20.1
// CHECK:STDOUT:   %.loc12_20.3: <unbound element of class Derived> = base_decl Base, element0
// CHECK:STDOUT:   %SelfBase: <function> = fn_decl @SelfBase
// CHECK:STDOUT:   %AddrSelfBase: <function> = fn_decl @AddrSelfBase
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12_20.3
// CHECK:STDOUT:   .SelfBase = %SelfBase
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase[%self: Base]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base = name_ref self, %self
// CHECK:STDOUT:   %.loc19_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_14.2: i32 = bind_value %.loc19_14.1
// CHECK:STDOUT:   return %.loc19_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase[addr %self: Base*]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base* = name_ref self, %self
// CHECK:STDOUT:   %.loc23_4: ref Base = deref %self.ref
// CHECK:STDOUT:   %.loc23_10: ref i32 = class_element_access %.loc23_4, element0
// CHECK:STDOUT:   %.loc23_15: i32 = int_literal 1
// CHECK:STDOUT:   assign %.loc23_10, %.loc23_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc27: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc27_4.1: ref Derived = deref %p.ref.loc27
// CHECK:STDOUT:   %.loc27_7: <bound method> = bound_method %.loc27_4.1, @Derived.%AddrSelfBase
// CHECK:STDOUT:   %.loc27_4.2: Derived* = address_of %.loc27_4.1
// CHECK:STDOUT:   %.loc27_20.1: ref Derived = deref %.loc27_4.2
// CHECK:STDOUT:   %.loc27_20.2: ref Base = class_element_access %.loc27_20.1, element0
// CHECK:STDOUT:   %.loc27_20.3: Base* = address_of %.loc27_20.2
// CHECK:STDOUT:   %.loc27_20.4: Base* = converted %.loc27_4.2, %.loc27_20.3
// CHECK:STDOUT:   %.loc27_20.5: init () = call %.loc27_7(%.loc27_20.4)
// CHECK:STDOUT:   %p.ref.loc28: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc28_11: ref Derived = deref %p.ref.loc28
// CHECK:STDOUT:   %.loc28_14: <bound method> = bound_method %.loc28_11, @Derived.%SelfBase
// CHECK:STDOUT:   %.loc28_23.1: ref Base = class_element_access %.loc28_11, element0
// CHECK:STDOUT:   %.loc28_23.2: ref Base = converted %.loc28_11, %.loc28_23.1
// CHECK:STDOUT:   %.loc28_23.3: Base = bind_value %.loc28_23.2
// CHECK:STDOUT:   %.loc28_23.4: init i32 = call %.loc28_14(%.loc28_23.3)
// CHECK:STDOUT:   %.loc28_25.1: i32 = value_of_initializer %.loc28_23.4
// CHECK:STDOUT:   %.loc28_25.2: i32 = converted %.loc28_23.4, %.loc28_25.1
// CHECK:STDOUT:   return %.loc28_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
