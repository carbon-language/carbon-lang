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
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Derived, Base [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %G.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %struct.4: G = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: Base} [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.5: Call = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: {}*} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.base: Base} [template]
// CHECK:STDOUT:   %.8: type = ptr_type Derived [template]
// CHECK:STDOUT:   %CallIndirect: type = fn_type @CallIndirect [template]
// CHECK:STDOUT:   %struct.6: CallIndirect = struct_value () [template]
// CHECK:STDOUT:   %PassDerivedToBase: type = fn_type @PassDerivedToBase [template]
// CHECK:STDOUT:   %struct.7: PassDerivedToBase = struct_value () [template]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect: type = fn_type @PassDerivedToBaseIndirect [template]
// CHECK:STDOUT:   %struct.8: PassDerivedToBaseIndirect = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Derived = %Derived.decl.loc7
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:     .PassDerivedToBase = %PassDerivedToBase.decl
// CHECK:STDOUT:     .PassDerivedToBaseIndirect = %PassDerivedToBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Derived.decl.loc7: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl.loc14: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.5] {
// CHECK:STDOUT:     %Derived.ref.loc21: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc21_9.1: Derived = param a
// CHECK:STDOUT:     @Call.%a: Derived = bind_name a, %a.loc21_9.1
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: CallIndirect = fn_decl @CallIndirect [template = constants.%struct.6] {
// CHECK:STDOUT:     %Derived.ref.loc25: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc25: type = ptr_type Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc25_17.1: Derived* = param p
// CHECK:STDOUT:     @CallIndirect.%p: Derived* = bind_name p, %p.loc25_17.1
// CHECK:STDOUT:     @CallIndirect.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: PassDerivedToBase = fn_decl @PassDerivedToBase [template = constants.%struct.7] {
// CHECK:STDOUT:     %Derived.ref.loc29: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc29_22.1: Derived = param a
// CHECK:STDOUT:     @PassDerivedToBase.%a: Derived = bind_name a, %a.loc29_22.1
// CHECK:STDOUT:     @PassDerivedToBase.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: PassDerivedToBaseIndirect = fn_decl @PassDerivedToBaseIndirect [template = constants.%struct.8] {
// CHECK:STDOUT:     %Derived.ref.loc33: type = name_ref Derived, %Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc33: type = ptr_type Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc33_30.1: Derived* = param p
// CHECK:STDOUT:     @PassDerivedToBaseIndirect.%p: Derived* = bind_name p, %p.loc33_30.1
// CHECK:STDOUT:     @PassDerivedToBaseIndirect.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc15: <unbound element of class Derived> = base_decl Base, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc17_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc17_8.2: Derived = bind_name self, %self.loc17_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.2 [template = constants.%struct.4] {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc18_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc18_8.2: Derived = bind_name self, %self.loc18_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc15
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.loc10_8.1: Base = param self
// CHECK:STDOUT:     %self.loc10_8.2: Base = bind_name self, %self.loc10_8.1
// CHECK:STDOUT:     %return.var.loc10: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc7 [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc11_8.1: Derived = param self
// CHECK:STDOUT:     %self.loc11_8.2: Derived = bind_name self, %self.loc11_8.1
// CHECK:STDOUT:     %return.var.loc11: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Base.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc22_11: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc22_20.1: ref Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc22_20.2: ref Base = converted %a.ref, %.loc22_20.1
// CHECK:STDOUT:   %.loc22_20.3: Base = bind_value %.loc22_20.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc22_11(%.loc22_20.3)
// CHECK:STDOUT:   %.loc22_22.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_22.2: i32 = converted %F.call, %.loc22_22.1
// CHECK:STDOUT:   return %.loc22_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Base.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc26_11.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc26_11.2: <bound method> = bound_method %.loc26_11.1, %F.ref
// CHECK:STDOUT:   %.loc26_21.1: ref Base = class_element_access %.loc26_11.1, element0
// CHECK:STDOUT:   %.loc26_21.2: ref Base = converted %.loc26_11.1, %.loc26_21.1
// CHECK:STDOUT:   %.loc26_21.3: Base = bind_value %.loc26_21.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc26_11.2(%.loc26_21.3)
// CHECK:STDOUT:   %.loc26_23.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_23.2: i32 = converted %F.call, %.loc26_23.1
// CHECK:STDOUT:   return %.loc26_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a: Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: G = name_ref G, @Base.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc30_11: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %G.call: init i32 = call %.loc30_11(%a.ref)
// CHECK:STDOUT:   %.loc30_22.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc30_22.2: i32 = converted %G.call, %.loc30_22.1
// CHECK:STDOUT:   return %.loc30_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p: Derived*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Derived* = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: G = name_ref G, @Base.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc34_11.1: ref Derived = deref %p.ref
// CHECK:STDOUT:   %.loc34_11.2: <bound method> = bound_method %.loc34_11.1, %G.ref
// CHECK:STDOUT:   %.loc34_11.3: Derived = bind_value %.loc34_11.1
// CHECK:STDOUT:   %G.call: init i32 = call %.loc34_11.2(%.loc34_11.3)
// CHECK:STDOUT:   %.loc34_23.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc34_23.2: i32 = converted %G.call, %.loc34_23.1
// CHECK:STDOUT:   return %.loc34_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
