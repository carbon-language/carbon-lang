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
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Base, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %SelfBase: type = fn_type @SelfBase [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: SelfBase = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type Base [template]
// CHECK:STDOUT:   %AddrSelfBase: type = fn_type @AddrSelfBase [template]
// CHECK:STDOUT:   %struct.2: AddrSelfBase = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: type = ptr_type Derived [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.3: Call = struct_value () [template]
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
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %SelfBase.decl: SelfBase = fn_decl @SelfBase [template = constants.%struct.1] {
// CHECK:STDOUT:     %Base.ref.loc18: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc18_21.1: Base = param self
// CHECK:STDOUT:     @SelfBase.%self: Base = bind_name self, %self.loc18_21.1
// CHECK:STDOUT:     @SelfBase.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: AddrSelfBase = fn_decl @AddrSelfBase [template = constants.%struct.2] {
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc22_40: type = ptr_type Base [template = constants.%.6]
// CHECK:STDOUT:     %self.loc22_30.1: Base* = param self
// CHECK:STDOUT:     @AddrSelfBase.%self: Base* = bind_name self, %self.loc22_30.1
// CHECK:STDOUT:     @AddrSelfBase.%.loc22: Base* = addr_pattern @AddrSelfBase.%self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.3] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc26: type = ptr_type Derived [template = constants.%.9]
// CHECK:STDOUT:     %p.loc26_9.1: Derived* = param p
// CHECK:STDOUT:     @Call.%p: Derived* = bind_name p, %p.loc26_9.1
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8: <unbound element of class Base> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref.loc12: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc12: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %SelfBase.decl: SelfBase = fn_decl @SelfBase [template = constants.%struct.1] {
// CHECK:STDOUT:     %Base.ref.loc14: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc14_15.1: Base = param self
// CHECK:STDOUT:     %self.loc14_15.2: Base = bind_name self, %self.loc14_15.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: AddrSelfBase = fn_decl @AddrSelfBase [template = constants.%struct.2] {
// CHECK:STDOUT:     %Base.ref.loc15: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc15_34: type = ptr_type Base [template = constants.%.6]
// CHECK:STDOUT:     %self.loc15_24.1: Base* = param self
// CHECK:STDOUT:     %self.loc15_24.3: Base* = bind_name self, %self.loc15_24.1
// CHECK:STDOUT:     %.loc15_19: Base* = addr_pattern %self.loc15_24.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase[%self: Base]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base = name_ref self, %self
// CHECK:STDOUT:   %a.ref: <unbound element of class Base> = name_ref a, @Base.%.loc8 [template = @Base.%.loc8]
// CHECK:STDOUT:   %.loc19_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_14.2: i32 = bind_value %.loc19_14.1
// CHECK:STDOUT:   return %.loc19_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase[addr %self: Base*]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Base* = name_ref self, %self
// CHECK:STDOUT:   %.loc23_4: ref Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: <unbound element of class Base> = name_ref a, @Base.%.loc8 [template = @Base.%.loc8]
// CHECK:STDOUT:   %.loc23_10: ref i32 = class_element_access %.loc23_4, element0
// CHECK:STDOUT:   %.loc23_15: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   assign %.loc23_10, %.loc23_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc27: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc27_4.1: ref Derived = deref %p.ref.loc27
// CHECK:STDOUT:   %AddrSelfBase.ref: AddrSelfBase = name_ref AddrSelfBase, @Derived.%AddrSelfBase.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc27_7: <bound method> = bound_method %.loc27_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %.loc27_4.2: Derived* = addr_of %.loc27_4.1
// CHECK:STDOUT:   %.loc27_20.1: ref Derived = deref %.loc27_4.2
// CHECK:STDOUT:   %.loc27_20.2: ref Base = class_element_access %.loc27_20.1, element0
// CHECK:STDOUT:   %.loc27_20.3: Base* = addr_of %.loc27_20.2
// CHECK:STDOUT:   %.loc27_20.4: Base* = converted %.loc27_4.2, %.loc27_20.3
// CHECK:STDOUT:   %AddrSelfBase.call: init () = call %.loc27_7(%.loc27_20.4)
// CHECK:STDOUT:   %p.ref.loc28: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc28_11: ref Derived = deref %p.ref.loc28
// CHECK:STDOUT:   %SelfBase.ref: SelfBase = name_ref SelfBase, @Derived.%SelfBase.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc28_14: <bound method> = bound_method %.loc28_11, %SelfBase.ref
// CHECK:STDOUT:   %.loc28_23.1: ref Base = class_element_access %.loc28_11, element0
// CHECK:STDOUT:   %.loc28_23.2: ref Base = converted %.loc28_11, %.loc28_23.1
// CHECK:STDOUT:   %.loc28_23.3: Base = bind_value %.loc28_23.2
// CHECK:STDOUT:   %SelfBase.call: init i32 = call %.loc28_14(%.loc28_23.3)
// CHECK:STDOUT:   %.loc28_25.1: i32 = value_of_initializer %SelfBase.call
// CHECK:STDOUT:   %.loc28_25.2: i32 = converted %SelfBase.call, %.loc28_25.1
// CHECK:STDOUT:   return %.loc28_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
