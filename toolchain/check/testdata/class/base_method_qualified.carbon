// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Derived;

base class Base {
  fn F[self: Self]() -> i32;
  fn G[self: Derived]() -> i32;
}

class Derived {
  extend base: Base;

  fn F[self: Self]();
  fn G[self: Self]();
}

fn Call(a: Derived) -> i32 {
  return a.(Base.F)();
}

fn CallIndirect(p: Derived*) -> i32 {
  return p->(Base.F)();
}

fn PassDerivedToBase(a: Derived) -> i32 {
  return a.(Base.G)();
}

fn PassDerivedToBaseIndirect(p: Derived*) -> i32 {
  return p->(Base.G)();
}

// CHECK:STDOUT: --- base_method_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: {}*} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.8: type = ptr_type Derived [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Derived = %Derived.decl.loc7
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Call = %Call
// CHECK:STDOUT:     .CallIndirect = %CallIndirect
// CHECK:STDOUT:     .PassDerivedToBase = %PassDerivedToBase
// CHECK:STDOUT:     .PassDerivedToBaseIndirect = %PassDerivedToBaseIndirect
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl.loc7: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl.loc14: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call [template] {
// CHECK:STDOUT:     %Derived.ref.loc21: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc21_9.1: Derived = param a
// CHECK:STDOUT:     @Call.%a: Derived = bind_name a, %a.loc21_9.1
// CHECK:STDOUT:     %return.var.loc21: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect: <function> = fn_decl @CallIndirect [template] {
// CHECK:STDOUT:     %Derived.ref.loc25: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc25: type = ptr_type Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc25_17.1: Derived* = param p
// CHECK:STDOUT:     @CallIndirect.%p: Derived* = bind_name p, %p.loc25_17.1
// CHECK:STDOUT:     %return.var.loc25: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase: <function> = fn_decl @PassDerivedToBase [template] {
// CHECK:STDOUT:     %Derived.ref.loc29: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc29_22.1: Derived = param a
// CHECK:STDOUT:     @PassDerivedToBase.%a: Derived = bind_name a, %a.loc29_22.1
// CHECK:STDOUT:     %return.var.loc29: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect: <function> = fn_decl @PassDerivedToBaseIndirect [template] {
// CHECK:STDOUT:     %Derived.ref.loc33: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc33: type = ptr_type Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc33_30.1: Derived* = param p
// CHECK:STDOUT:     @PassDerivedToBaseIndirect.%p: Derived* = bind_name p, %p.loc33_30.1
// CHECK:STDOUT:     %return.var.loc33: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc15: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc17_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc17_8.2: Derived = bind_name self, %self.loc17_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc18_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc18_8.2: Derived = bind_name self, %self.loc18_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.loc10_8.1: Base = param self
// CHECK:STDOUT:     %self.loc10_8.2: Base = bind_name self, %self.loc10_8.1
// CHECK:STDOUT:     %return.var.loc10: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G.1 [template] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc11_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc11_8.2: Derived = bind_name self, %self.loc11_8.1
// CHECK:STDOUT:     %return.var.loc11: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@Base.%self.loc10_8.2: Base]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[@Base.%self.loc11_8.2: Derived]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@Derived.%self.loc17_8.2: Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@Derived.%self.loc18_8.2: Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a: Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F [template = @Base.%F]
// CHECK:STDOUT:   %.loc22_11: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc22_20.1: ref Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc22_10.1: ref Base = converted %a.ref, %.loc22_20.1
// CHECK:STDOUT:   %.loc22_10.2: Base = bind_value %.loc22_10.1
// CHECK:STDOUT:   %.loc22_20.2: init i32 = call %.loc22_11(%.loc22_10.2)
// CHECK:STDOUT:   %.loc22_22: i32 = value_of_initializer %.loc22_20.2
// CHECK:STDOUT:   %.loc22_20.3: i32 = converted %.loc22_20.2, %.loc22_22
// CHECK:STDOUT:   return %.loc22_20.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Base.%F [template = @Base.%F]
// CHECK:STDOUT:   %.loc26_11.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc26_11.2: <bound method> = bound_method %.loc26_11.1, %F.ref
// CHECK:STDOUT:   %.loc26_21.1: ref Base = class_element_access %.loc26_11.1, element0
// CHECK:STDOUT:   %.loc26_11.3: ref Base = converted %.loc26_11.1, %.loc26_21.1
// CHECK:STDOUT:   %.loc26_11.4: Base = bind_value %.loc26_11.3
// CHECK:STDOUT:   %.loc26_21.2: init i32 = call %.loc26_11.2(%.loc26_11.4)
// CHECK:STDOUT:   %.loc26_23: i32 = value_of_initializer %.loc26_21.2
// CHECK:STDOUT:   %.loc26_21.3: i32 = converted %.loc26_21.2, %.loc26_23
// CHECK:STDOUT:   return %.loc26_21.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a: Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Base.%G [template = @Base.%G]
// CHECK:STDOUT:   %.loc30_11: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %.loc30_20.1: init i32 = call %.loc30_11(%a.ref)
// CHECK:STDOUT:   %.loc30_22: i32 = value_of_initializer %.loc30_20.1
// CHECK:STDOUT:   %.loc30_20.2: i32 = converted %.loc30_20.1, %.loc30_22
// CHECK:STDOUT:   return %.loc30_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Base.%G [template = @Base.%G]
// CHECK:STDOUT:   %.loc34_11.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc34_11.2: <bound method> = bound_method %.loc34_11.1, %G.ref
// CHECK:STDOUT:   %.loc34_11.3: Derived = bind_value %.loc34_11.1
// CHECK:STDOUT:   %.loc34_21.1: init i32 = call %.loc34_11.2(%.loc34_11.3)
// CHECK:STDOUT:   %.loc34_23: i32 = value_of_initializer %.loc34_21.1
// CHECK:STDOUT:   %.loc34_21.2: i32 = converted %.loc34_21.1, %.loc34_23
// CHECK:STDOUT:   return %.loc34_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
