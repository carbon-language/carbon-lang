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
// CHECK:STDOUT:   %.5: type = ptr_type Base [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: type = ptr_type Derived [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: {.a: i32}*} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.11: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %SelfBase: <function> = fn_decl @SelfBase [template] {
// CHECK:STDOUT:     %Base.ref.loc18: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc18_21.1: Base = param self
// CHECK:STDOUT:     @SelfBase.%self: Base = bind_name self, %self.loc18_21.1
// CHECK:STDOUT:     %return.var.loc18: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase: <function> = fn_decl @AddrSelfBase [template] {
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, %Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc22_40: type = ptr_type Base [template = constants.%.5]
// CHECK:STDOUT:     %self.loc22_30.1: Base* = param self
// CHECK:STDOUT:     @AddrSelfBase.%self: Base* = bind_name self, %self.loc22_30.1
// CHECK:STDOUT:     @AddrSelfBase.%.loc22: Base* = addr_pattern @AddrSelfBase.%self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, %Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %.loc26: type = ptr_type Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc26_9.1: Derived* = param p
// CHECK:STDOUT:     @Call.%p: Derived* = bind_name p, %p.loc26_9.1
// CHECK:STDOUT:     %return.var.loc26: ref i32 = var <return slot>
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
// CHECK:STDOUT:   %SelfBase: <function> = fn_decl @SelfBase [template] {
// CHECK:STDOUT:     %Base.ref.loc14: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %self.loc14_15.1: Base = param self
// CHECK:STDOUT:     %self.loc14_15.2: Base = bind_name self, %self.loc14_15.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase: <function> = fn_decl @AddrSelfBase [template] {
// CHECK:STDOUT:     %Base.ref.loc15: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:     %.loc15_34: type = ptr_type Base [template = constants.%.5]
// CHECK:STDOUT:     %self.loc15_24.1: Base* = param self
// CHECK:STDOUT:     %self.loc15_24.3: Base* = bind_name self, %self.loc15_24.1
// CHECK:STDOUT:     %.loc15_19: Base* = addr_pattern %self.loc15_24.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .SelfBase = %SelfBase
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase
// CHECK:STDOUT:   extend name_scope1
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
// CHECK:STDOUT:   %.loc23_15: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   assign %.loc23_10, %.loc23_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc27: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc27_4.1: ref Derived = deref %p.ref.loc27
// CHECK:STDOUT:   %AddrSelfBase.ref: <function> = name_ref AddrSelfBase, @Derived.%AddrSelfBase [template = @Derived.%AddrSelfBase]
// CHECK:STDOUT:   %.loc27_7: <bound method> = bound_method %.loc27_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %.loc27_4.2: Derived* = addr_of %.loc27_4.1
// CHECK:STDOUT:   %.loc27_20.1: ref Derived = deref %.loc27_4.2
// CHECK:STDOUT:   %.loc27_20.2: ref Base = class_element_access %.loc27_20.1, element0
// CHECK:STDOUT:   %.loc27_20.3: Base* = addr_of %.loc27_20.2
// CHECK:STDOUT:   %.loc27_4.3: Base* = converted %.loc27_4.2, %.loc27_20.3
// CHECK:STDOUT:   %.loc27_20.4: init () = call %.loc27_7(%.loc27_4.3)
// CHECK:STDOUT:   %p.ref.loc28: Derived* = name_ref p, %p
// CHECK:STDOUT:   %.loc28_11.1: ref Derived = deref %p.ref.loc28
// CHECK:STDOUT:   %SelfBase.ref: <function> = name_ref SelfBase, @Derived.%SelfBase [template = @Derived.%SelfBase]
// CHECK:STDOUT:   %.loc28_14: <bound method> = bound_method %.loc28_11.1, %SelfBase.ref
// CHECK:STDOUT:   %.loc28_23.1: ref Base = class_element_access %.loc28_11.1, element0
// CHECK:STDOUT:   %.loc28_11.2: ref Base = converted %.loc28_11.1, %.loc28_23.1
// CHECK:STDOUT:   %.loc28_11.3: Base = bind_value %.loc28_11.2
// CHECK:STDOUT:   %.loc28_23.2: init i32 = call %.loc28_14(%.loc28_11.3)
// CHECK:STDOUT:   %.loc28_25: i32 = value_of_initializer %.loc28_23.2
// CHECK:STDOUT:   %.loc28_23.3: i32 = converted %.loc28_23.2, %.loc28_25
// CHECK:STDOUT:   return %.loc28_23.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
