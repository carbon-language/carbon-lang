// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Eq {
  fn Equal[self: Self](other: Self) -> bool;
  fn NotEqual[self: Self](other: Self) -> bool;
}

// --- user.carbon

package User api;

import Core;

class C {};

impl C as Core.Eq {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestEqual(a: C, b: C) -> bool {
  return a == b;
}

fn TestNotEqual(a: C, b: C) -> bool {
  return a != b;
}

// --- fail_no_impl.carbon

package FailNoImpl api;

import Core;

class D {};

fn TestEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Eq in type D that does not implement that interface.
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestNotEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Eq in type D that does not implement that interface.
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a != b;
}

// --- fail_no_impl_for_args.carbon

package FailNoImplForArgs api;

import Core;

class C {};
class D {};

impl C as Core.Eq {
  fn Equal[self: C](other: C) -> bool;
  fn NotEqual[self: C](other: C) -> bool;
}

fn TestRhsBad(a: C, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+7]]:10: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   return a == b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE-8]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Equal[self: C](other: C) -> bool;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return a == b;
}

fn TestLhsBad(a: D, b: C) -> bool {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Eq in type D that does not implement that interface.
  // CHECK:STDERR:   return a != b;
  // CHECK:STDERR:          ^~~~~~
  return a != b;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Eq [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Eq, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Eq> = assoc_entity element0, @Eq.%Equal [template]
// CHECK:STDOUT:   %.4: <associated <function> in Eq> = assoc_entity element1, @Eq.%NotEqual [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eq.decl: type = interface_decl @Eq [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT:   %Self: Eq = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Equal: <function> = fn_decl @Equal [template] {
// CHECK:STDOUT:     %Self.ref.loc5_18: Eq = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_18.1: type = facet_type_access %Self.ref.loc5_18 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_18.2: type = converted %Self.ref.loc5_18, %.loc5_18.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc5_12.1: Self = param self
// CHECK:STDOUT:     %self.loc5_12.2: Self = bind_name self, %self.loc5_12.1
// CHECK:STDOUT:     %Self.ref.loc5_31: Eq = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_31.1: type = facet_type_access %Self.ref.loc5_31 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %Self.ref.loc5_31, %.loc5_31.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc5_24.1: Self = param other
// CHECK:STDOUT:     %other.loc5_24.2: Self = bind_name other, %other.loc5_24.1
// CHECK:STDOUT:     %return.var.loc5: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_44: <associated <function> in Eq> = assoc_entity element0, %Equal [template = constants.%.3]
// CHECK:STDOUT:   %NotEqual: <function> = fn_decl @NotEqual [template] {
// CHECK:STDOUT:     %Self.ref.loc6_21: Eq = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_21.1: type = facet_type_access %Self.ref.loc6_21 [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_21.2: type = converted %Self.ref.loc6_21, %.loc6_21.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc6_15.1: Self = param self
// CHECK:STDOUT:     %self.loc6_15.2: Self = bind_name self, %self.loc6_15.1
// CHECK:STDOUT:     %Self.ref.loc6_34: Eq = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_34.1: type = facet_type_access %Self.ref.loc6_34 [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %Self.ref.loc6_34, %.loc6_34.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc6_27.1: Self = param other
// CHECK:STDOUT:     %other.loc6_27.2: Self = bind_name other, %other.loc6_27.1
// CHECK:STDOUT:     %return.var.loc6: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_47: <associated <function> in Eq> = assoc_entity element1, %NotEqual [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Equal = %.loc5_44
// CHECK:STDOUT:   .NotEqual = %.loc6_47
// CHECK:STDOUT:   witness = (%Equal, %NotEqual)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal[@Eq.%self.loc5_12.2: Self](@Eq.%other.loc5_24.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual[@Eq.%self.loc6_15.2: Self](@Eq.%other.loc6_27.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (@impl.%Equal, @impl.%NotEqual) [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Eq, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in Eq> = assoc_entity element0, file.%import_ref.8 [template]
// CHECK:STDOUT:   %.8: <associated <function> in Eq> = assoc_entity element1, file.%import_ref.10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Eq> = import_ref ir1, inst+17, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <associated <function> in Eq> = import_ref ir1, inst+31, used [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.5: <function> = import_ref ir1, inst+15, used [template = imports.%Equal]
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+30, used [template = imports.%NotEqual]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Eq.decl: invalid = interface_decl @Eq [template = constants.%.2] {}
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestEqual: <function> = fn_decl @TestEqual [template] {
// CHECK:STDOUT:     %C.ref.loc13_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc13_14.1: C = param a
// CHECK:STDOUT:     @TestEqual.%a: C = bind_name a, %a.loc13_14.1
// CHECK:STDOUT:     %C.ref.loc13_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc13_20.1: C = param b
// CHECK:STDOUT:     @TestEqual.%b: C = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %return.var.loc13: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %TestNotEqual: <function> = fn_decl @TestNotEqual [template] {
// CHECK:STDOUT:     %C.ref.loc17_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc17_17.1: C = param a
// CHECK:STDOUT:     @TestNotEqual.%a: C = bind_name a, %a.loc17_17.1
// CHECK:STDOUT:     %C.ref.loc17_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc17_23.1: C = param b
// CHECK:STDOUT:     @TestNotEqual.%b: C = bind_name b, %b.loc17_23.1
// CHECK:STDOUT:     %return.var.loc17: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+30, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .NotEqual = file.%import_ref.4
// CHECK:STDOUT:   witness = (file.%import_ref.5, file.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Eq {
// CHECK:STDOUT:   %Equal: <function> = fn_decl @Equal.1 [template] {
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_12.1: C = param self
// CHECK:STDOUT:     %self.loc9_12.2: C = bind_name self, %self.loc9_12.1
// CHECK:STDOUT:     %C.ref.loc9_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_21.1: C = param other
// CHECK:STDOUT:     %other.loc9_21.2: C = bind_name other, %other.loc9_21.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual: <function> = fn_decl @NotEqual.1 [template] {
// CHECK:STDOUT:     %C.ref.loc10_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc10_15.1: C = param self
// CHECK:STDOUT:     %self.loc10_15.2: C = bind_name self, %self.loc10_15.1
// CHECK:STDOUT:     %C.ref.loc10_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc10_24.1: C = param other
// CHECK:STDOUT:     %other.loc10_24.2: C = bind_name other, %other.loc10_24.1
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Equal, %NotEqual) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal
// CHECK:STDOUT:   .NotEqual = %NotEqual
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc9_12.2: C](@impl.%other.loc9_21.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc10_15.2: C](@impl.%other.loc10_24.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Equal]
// CHECK:STDOUT:   %.loc14_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc14_12.2: init bool = call %.loc14_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc14_16: bool = value_of_initializer %.loc14_12.2
// CHECK:STDOUT:   %.loc14_12.3: bool = converted %.loc14_12.2, %.loc14_16
// CHECK:STDOUT:   return %.loc14_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element1 [template = @impl.%NotEqual]
// CHECK:STDOUT:   %.loc18_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc18_12.2: init bool = call %.loc18_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc18_16: bool = value_of_initializer %.loc18_12.2
// CHECK:STDOUT:   %.loc18_12.3: bool = converted %.loc18_12.2, %.loc18_16
// CHECK:STDOUT:   return %.loc18_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Eq [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Eq, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Eq> = assoc_entity element0, file.%import_ref.7 [template]
// CHECK:STDOUT:   %.7: <associated <function> in Eq> = assoc_entity element1, file.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestEqual: <function> = fn_decl @TestEqual [template] {
// CHECK:STDOUT:     %D.ref.loc8_17: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc8_14.1: D = param a
// CHECK:STDOUT:     @TestEqual.%a: D = bind_name a, %a.loc8_14.1
// CHECK:STDOUT:     %D.ref.loc8_23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc8_20.1: D = param b
// CHECK:STDOUT:     @TestEqual.%b: D = bind_name b, %b.loc8_20.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Eq> = import_ref ir1, inst+17, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <associated <function> in Eq> = import_ref ir1, inst+31, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.5 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %TestNotEqual: <function> = fn_decl @TestNotEqual [template] {
// CHECK:STDOUT:     %D.ref.loc16_20: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc16_17.1: D = param a
// CHECK:STDOUT:     @TestNotEqual.%a: D = bind_name a, %a.loc16_17.1
// CHECK:STDOUT:     %D.ref.loc16_26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc16_23.1: D = param b
// CHECK:STDOUT:     @TestNotEqual.%b: D = bind_name b, %b.loc16_23.1
// CHECK:STDOUT:     %return.var.loc16: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+30, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .NotEqual = file.%import_ref.4
// CHECK:STDOUT:   witness = (file.%import_ref.5, file.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %Eq.decl: invalid = interface_decl @Eq [template = constants.%.4] {}
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl_for_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (@impl.%Equal, @impl.%NotEqual) [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Eq, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in Eq> = assoc_entity element0, file.%import_ref.8 [template]
// CHECK:STDOUT:   %.8: <associated <function> in Eq> = assoc_entity element1, file.%import_ref.10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestRhsBad = %TestRhsBad
// CHECK:STDOUT:     .TestLhsBad = %TestLhsBad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Eq> = import_ref ir1, inst+17, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <associated <function> in Eq> = import_ref ir1, inst+31, used [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.5: <function> = import_ref ir1, inst+15, used [template = imports.%Equal]
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+30, used [template = imports.%NotEqual]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Eq.decl: invalid = interface_decl @Eq [template = constants.%.2] {}
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRhsBad: <function> = fn_decl @TestRhsBad [template] {
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc14_15.1: C = param a
// CHECK:STDOUT:     @TestRhsBad.%a: C = bind_name a, %a.loc14_15.1
// CHECK:STDOUT:     %D.ref.loc14: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc14_21.1: D = param b
// CHECK:STDOUT:     @TestRhsBad.%b: D = bind_name b, %b.loc14_21.1
// CHECK:STDOUT:     %return.var.loc14: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %TestLhsBad: <function> = fn_decl @TestLhsBad [template] {
// CHECK:STDOUT:     %D.ref.loc25: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc25_15.1: D = param a
// CHECK:STDOUT:     @TestLhsBad.%a: D = bind_name a, %a.loc25_15.1
// CHECK:STDOUT:     %C.ref.loc25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc25_21.1: C = param b
// CHECK:STDOUT:     @TestLhsBad.%b: C = bind_name b, %b.loc25_21.1
// CHECK:STDOUT:     %return.var.loc25: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+30, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .NotEqual = file.%import_ref.4
// CHECK:STDOUT:   witness = (file.%import_ref.5, file.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Eq {
// CHECK:STDOUT:   %Equal: <function> = fn_decl @Equal.1 [template] {
// CHECK:STDOUT:     %C.ref.loc10_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc10_12.1: C = param self
// CHECK:STDOUT:     %self.loc10_12.2: C = bind_name self, %self.loc10_12.1
// CHECK:STDOUT:     %C.ref.loc10_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc10_21.1: C = param other
// CHECK:STDOUT:     %other.loc10_21.2: C = bind_name other, %other.loc10_21.1
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual: <function> = fn_decl @NotEqual.1 [template] {
// CHECK:STDOUT:     %C.ref.loc11_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc11_15.1: C = param self
// CHECK:STDOUT:     %self.loc11_15.2: C = bind_name self, %self.loc11_15.1
// CHECK:STDOUT:     %C.ref.loc11_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc11_24.1: C = param other
// CHECK:STDOUT:     %other.loc11_24.2: C = bind_name other, %other.loc11_24.1
// CHECK:STDOUT:     %return.var.loc11: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Equal, %NotEqual) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal
// CHECK:STDOUT:   .NotEqual = %NotEqual
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc10_12.2: C](@impl.%other.loc10_21.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc11_15.2: C](@impl.%other.loc11_24.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a: C, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Equal]
// CHECK:STDOUT:   %.loc22_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc22_12.2: init bool = call %.loc22_12.1(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc22_16: bool = value_of_initializer %.loc22_12.2 [template = <error>]
// CHECK:STDOUT:   %.loc22_12.3: bool = converted %.loc22_12.2, %.loc22_16 [template = <error>]
// CHECK:STDOUT:   return %.loc22_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a: D, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
