// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- user.carbon

package User;

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

package FailNoImpl;

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

package FailNoImplForArgs;

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

// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self: Eq = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Equal.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Equal = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %struct.2: NotEqual = struct_value () [template]
// CHECK:STDOUT:   %Equal.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %struct.3: Equal = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %struct.4: NotEqual = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%struct.1, %struct.2) [template]
// CHECK:STDOUT:   %TestEqual: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %struct.5: TestEqual = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Eq, Equal [template]
// CHECK:STDOUT:   %.7: <associated Equal in Eq> = assoc_entity element0, file.%import_ref.8 [template]
// CHECK:STDOUT:   %TestNotEqual: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %struct.6: TestNotEqual = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Eq, NotEqual [template]
// CHECK:STDOUT:   %.9: <associated NotEqual in Eq> = assoc_entity element1, file.%import_ref.10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir6, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Equal in Eq> = import_ref ir6, inst+21, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: <associated NotEqual in Eq> = import_ref ir6, inst+38, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.5: Equal = import_ref ir6, inst+16, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.6: NotEqual = import_ref ir6, inst+34, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestEqual.decl: TestEqual = fn_decl @TestEqual [template = constants.%struct.5] {
// CHECK:STDOUT:     %C.ref.loc11_17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc11_14.1: C = param a
// CHECK:STDOUT:     @TestEqual.%a: C = bind_name a, %a.loc11_14.1
// CHECK:STDOUT:     %C.ref.loc11_23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc11_20.1: C = param b
// CHECK:STDOUT:     @TestEqual.%b: C = bind_name b, %b.loc11_20.1
// CHECK:STDOUT:     @TestEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir6, inst+16, unloaded
// CHECK:STDOUT:   %TestNotEqual.decl: TestNotEqual = fn_decl @TestNotEqual [template = constants.%struct.6] {
// CHECK:STDOUT:     %C.ref.loc15_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_17.1: C = param a
// CHECK:STDOUT:     @TestNotEqual.%a: C = bind_name a, %a.loc15_17.1
// CHECK:STDOUT:     %C.ref.loc15_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc15_23.1: C = param b
// CHECK:STDOUT:     @TestNotEqual.%b: C = bind_name b, %b.loc15_23.1
// CHECK:STDOUT:     @TestNotEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir6, inst+34, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Equal = file.%import_ref.3
// CHECK:STDOUT:   .NotEqual = file.%import_ref.4
// CHECK:STDOUT:   witness = (file.%import_ref.5, file.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Eq {
// CHECK:STDOUT:   %Equal.decl: Equal = fn_decl @Equal.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc7_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc7_12.1: C = param self
// CHECK:STDOUT:     %self.loc7_12.2: C = bind_name self, %self.loc7_12.1
// CHECK:STDOUT:     %C.ref.loc7_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc7_21.1: C = param other
// CHECK:STDOUT:     %other.loc7_21.2: C = bind_name other, %other.loc7_21.1
// CHECK:STDOUT:     %return.var.loc7: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: NotEqual = fn_decl @NotEqual.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc8_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_15.1: C = param self
// CHECK:STDOUT:     %self.loc8_15.2: C = bind_name self, %self.loc8_15.1
// CHECK:STDOUT:     %C.ref.loc8_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc8_24.1: C = param other
// CHECK:STDOUT:     %other.loc8_24.2: C = bind_name other, %other.loc8_24.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc7_12.2: C](@impl.%other.loc7_21.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc8_15.2: C](@impl.%other.loc8_24.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: Equal = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc12_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc12_16.1: bool = value_of_initializer %Equal.call
// CHECK:STDOUT:   %.loc12_16.2: bool = converted %Equal.call, %.loc12_16.1
// CHECK:STDOUT:   return %.loc12_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: NotEqual = interface_witness_access @impl.%.1, element1 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc16_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %NotEqual.call: init bool = call %.loc16_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_16.1: bool = value_of_initializer %NotEqual.call
// CHECK:STDOUT:   %.loc16_16.2: bool = converted %NotEqual.call, %.loc16_16.1
// CHECK:STDOUT:   return %.loc16_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %TestEqual: type = fn_type @TestEqual [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: TestEqual = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self: Eq = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Equal: type = fn_type @Equal [template]
// CHECK:STDOUT:   %struct.2: Equal = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Eq, Equal [template]
// CHECK:STDOUT:   %.6: <associated Equal in Eq> = assoc_entity element0, file.%import_ref.7 [template]
// CHECK:STDOUT:   %TestNotEqual: type = fn_type @TestNotEqual [template]
// CHECK:STDOUT:   %struct.3: TestNotEqual = struct_value () [template]
// CHECK:STDOUT:   %NotEqual: type = fn_type @NotEqual [template]
// CHECK:STDOUT:   %struct.4: NotEqual = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @Eq, NotEqual [template]
// CHECK:STDOUT:   %.8: <associated NotEqual in Eq> = assoc_entity element1, file.%import_ref.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestEqual = %TestEqual.decl
// CHECK:STDOUT:     .TestNotEqual = %TestNotEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestEqual.decl: TestEqual = fn_decl @TestEqual [template = constants.%struct.1] {
// CHECK:STDOUT:     %D.ref.loc6_17: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc6_14.1: D = param a
// CHECK:STDOUT:     @TestEqual.%a: D = bind_name a, %a.loc6_14.1
// CHECK:STDOUT:     %D.ref.loc6_23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc6_20.1: D = param b
// CHECK:STDOUT:     @TestEqual.%b: D = bind_name b, %b.loc6_20.1
// CHECK:STDOUT:     @TestEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir6, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir6, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Equal in Eq> = import_ref ir6, inst+21, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.4: <associated NotEqual in Eq> = import_ref ir6, inst+38, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.5 = import_ref ir6, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref ir6, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir6, inst+16, unloaded
// CHECK:STDOUT:   %TestNotEqual.decl: TestNotEqual = fn_decl @TestNotEqual [template = constants.%struct.3] {
// CHECK:STDOUT:     %D.ref.loc14_20: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc14_17.1: D = param a
// CHECK:STDOUT:     @TestNotEqual.%a: D = bind_name a, %a.loc14_17.1
// CHECK:STDOUT:     %D.ref.loc14_26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc14_23.1: D = param b
// CHECK:STDOUT:     @TestNotEqual.%b: D = bind_name b, %b.loc14_23.1
// CHECK:STDOUT:     @TestNotEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8: type = import_ref ir6, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.9 = import_ref ir6, inst+34, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Equal = file.%import_ref.3
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
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestNotEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl_for_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.2: type = interface_type @Eq [template]
// CHECK:STDOUT:   %Self: Eq = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Equal.1: type = fn_type @Equal.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Equal = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.1: type = fn_type @NotEqual.1 [template]
// CHECK:STDOUT:   %struct.2: NotEqual = struct_value () [template]
// CHECK:STDOUT:   %Equal.2: type = fn_type @Equal.2 [template]
// CHECK:STDOUT:   %struct.3: Equal = struct_value () [template]
// CHECK:STDOUT:   %NotEqual.2: type = fn_type @NotEqual.2 [template]
// CHECK:STDOUT:   %struct.4: NotEqual = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%struct.1, %struct.2) [template]
// CHECK:STDOUT:   %TestRhsBad: type = fn_type @TestRhsBad [template]
// CHECK:STDOUT:   %struct.5: TestRhsBad = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Eq, Equal [template]
// CHECK:STDOUT:   %.7: <associated Equal in Eq> = assoc_entity element0, file.%import_ref.8 [template]
// CHECK:STDOUT:   %TestLhsBad: type = fn_type @TestLhsBad [template]
// CHECK:STDOUT:   %struct.6: TestLhsBad = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Eq, NotEqual [template]
// CHECK:STDOUT:   %.9: <associated NotEqual in Eq> = assoc_entity element1, file.%import_ref.10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestRhsBad = %TestRhsBad.decl
// CHECK:STDOUT:     .TestLhsBad = %TestLhsBad.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir6, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Equal in Eq> = import_ref ir6, inst+21, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: <associated NotEqual in Eq> = import_ref ir6, inst+38, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.5: Equal = import_ref ir6, inst+16, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.6: NotEqual = import_ref ir6, inst+34, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Eq.ref: type = name_ref Eq, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestRhsBad.decl: TestRhsBad = fn_decl @TestRhsBad [template = constants.%struct.5] {
// CHECK:STDOUT:     %C.ref.loc12: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc12_15.1: C = param a
// CHECK:STDOUT:     @TestRhsBad.%a: C = bind_name a, %a.loc12_15.1
// CHECK:STDOUT:     %D.ref.loc12: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc12_21.1: D = param b
// CHECK:STDOUT:     @TestRhsBad.%b: D = bind_name b, %b.loc12_21.1
// CHECK:STDOUT:     @TestRhsBad.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.8 = import_ref ir6, inst+16, unloaded
// CHECK:STDOUT:   %TestLhsBad.decl: TestLhsBad = fn_decl @TestLhsBad [template = constants.%struct.6] {
// CHECK:STDOUT:     %D.ref.loc23: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc23_15.1: D = param a
// CHECK:STDOUT:     @TestLhsBad.%a: D = bind_name a, %a.loc23_15.1
// CHECK:STDOUT:     %C.ref.loc23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc23_21.1: C = param b
// CHECK:STDOUT:     @TestLhsBad.%b: C = bind_name b, %b.loc23_21.1
// CHECK:STDOUT:     @TestLhsBad.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir6, inst+34, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Eq {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Equal = file.%import_ref.3
// CHECK:STDOUT:   .NotEqual = file.%import_ref.4
// CHECK:STDOUT:   witness = (file.%import_ref.5, file.%import_ref.6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Eq {
// CHECK:STDOUT:   %Equal.decl: Equal = fn_decl @Equal.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc8_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_12.1: C = param self
// CHECK:STDOUT:     %self.loc8_12.2: C = bind_name self, %self.loc8_12.1
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc8_21.1: C = param other
// CHECK:STDOUT:     %other.loc8_21.2: C = bind_name other, %other.loc8_21.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotEqual.decl: NotEqual = fn_decl @NotEqual.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc9_21: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_15.1: C = param self
// CHECK:STDOUT:     %self.loc9_15.2: C = bind_name self, %self.loc9_15.1
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_24.1: C = param other
// CHECK:STDOUT:     %other.loc9_24.2: C = bind_name other, %other.loc9_24.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Equal.decl, %NotEqual.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Equal = %Equal.decl
// CHECK:STDOUT:   .NotEqual = %NotEqual.decl
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
// CHECK:STDOUT: fn @Equal.1[@impl.%self.loc8_12.2: C](@impl.%other.loc8_21.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.1[@impl.%self.loc9_15.2: C](@impl.%other.loc9_24.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Equal.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotEqual.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a: C, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %.1: Equal = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Equal.call: init bool = call %.loc20_12(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc20_16.1: bool = value_of_initializer %Equal.call [template = <error>]
// CHECK:STDOUT:   %.loc20_16.2: bool = converted %Equal.call, %.loc20_16.1 [template = <error>]
// CHECK:STDOUT:   return %.loc20_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a: D, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
