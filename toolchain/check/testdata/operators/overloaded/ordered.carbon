// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Ordered {
  // TODO: fn Compare
  fn Less[self: Self](other: Self) -> bool;
  fn LessOrEquivalent[self: Self](other: Self) -> bool;
  fn Greater[self: Self](other: Self) -> bool;
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool;
}

// --- user.carbon

package User api;

import Core;

class C {};

impl C as Core.Ordered {
  fn Less[self: C](other: C) -> bool;
  fn LessOrEquivalent[self: C](other: C) -> bool;
  fn Greater[self: C](other: C) -> bool;
  fn GreaterOrEquivalent[self: C](other: C) -> bool;
}

fn TestLess(a: C, b: C) -> bool {
  return a < b;
}

fn TestLessEqual(a: C, b: C) -> bool {
  return a <= b;
}

fn TestGreater(a: C, b: C) -> bool {
  return a > b;
}

fn TestGreaterEqual(a: C, b: C) -> bool {
  return a >= b;
}

// --- fail_no_impl.carbon

package FailNoImpl api;

import Core;

class D {};

fn TestLess(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Ordered in type D that does not implement that interface.
  // CHECK:STDERR:   return a < b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a < b;
}

fn TestLessEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Ordered in type D that does not implement that interface.
  // CHECK:STDERR:   return a <= b;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR:
  return a <= b;
}

fn TestGreater(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+4]]:10: ERROR: Cannot access member of interface Ordered in type D that does not implement that interface.
  // CHECK:STDERR:   return a > b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a > b;
}

fn TestGreaterEqual(a: D, b: D) -> bool {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Ordered in type D that does not implement that interface.
  // CHECK:STDERR:   return a >= b;
  // CHECK:STDERR:          ^~~~~~
  return a >= b;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Ordered, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Ordered> = assoc_entity element0, @Ordered.%Less [template]
// CHECK:STDOUT:   %.4: <associated <function> in Ordered> = assoc_entity element1, @Ordered.%LessOrEquivalent [template]
// CHECK:STDOUT:   %.5: <associated <function> in Ordered> = assoc_entity element2, @Ordered.%Greater [template]
// CHECK:STDOUT:   %.6: <associated <function> in Ordered> = assoc_entity element3, @Ordered.%GreaterOrEquivalent [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Ordered = %Ordered.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ordered.decl: type = interface_decl @Ordered [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT:   %Self: Ordered = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Less: <function> = fn_decl @Less [template] {
// CHECK:STDOUT:     %Self.ref.loc6_17: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_17.1: type = facet_type_access %Self.ref.loc6_17 [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_17.2: type = converted %Self.ref.loc6_17, %.loc6_17.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc6_11.1: Self = param self
// CHECK:STDOUT:     %self.loc6_11.2: Self = bind_name self, %self.loc6_11.1
// CHECK:STDOUT:     %Self.ref.loc6_30: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_30.1: type = facet_type_access %Self.ref.loc6_30 [symbolic = %Self]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %Self.ref.loc6_30, %.loc6_30.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc6_23.1: Self = param other
// CHECK:STDOUT:     %other.loc6_23.2: Self = bind_name other, %other.loc6_23.1
// CHECK:STDOUT:     %return.var.loc6: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_43: <associated <function> in Ordered> = assoc_entity element0, %Less [template = constants.%.3]
// CHECK:STDOUT:   %LessOrEquivalent: <function> = fn_decl @LessOrEquivalent [template] {
// CHECK:STDOUT:     %Self.ref.loc7_29: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc7_29.1: type = facet_type_access %Self.ref.loc7_29 [symbolic = %Self]
// CHECK:STDOUT:     %.loc7_29.2: type = converted %Self.ref.loc7_29, %.loc7_29.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc7_23.1: Self = param self
// CHECK:STDOUT:     %self.loc7_23.2: Self = bind_name self, %self.loc7_23.1
// CHECK:STDOUT:     %Self.ref.loc7_42: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc7_42.1: type = facet_type_access %Self.ref.loc7_42 [symbolic = %Self]
// CHECK:STDOUT:     %.loc7_42.2: type = converted %Self.ref.loc7_42, %.loc7_42.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc7_35.1: Self = param other
// CHECK:STDOUT:     %other.loc7_35.2: Self = bind_name other, %other.loc7_35.1
// CHECK:STDOUT:     %return.var.loc7: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_55: <associated <function> in Ordered> = assoc_entity element1, %LessOrEquivalent [template = constants.%.4]
// CHECK:STDOUT:   %Greater: <function> = fn_decl @Greater [template] {
// CHECK:STDOUT:     %Self.ref.loc8_20: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_20.1: type = facet_type_access %Self.ref.loc8_20 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_20.2: type = converted %Self.ref.loc8_20, %.loc8_20.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc8_14.1: Self = param self
// CHECK:STDOUT:     %self.loc8_14.2: Self = bind_name self, %self.loc8_14.1
// CHECK:STDOUT:     %Self.ref.loc8_33: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_33.1: type = facet_type_access %Self.ref.loc8_33 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_33.2: type = converted %Self.ref.loc8_33, %.loc8_33.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_26.1: Self = param other
// CHECK:STDOUT:     %other.loc8_26.2: Self = bind_name other, %other.loc8_26.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_46: <associated <function> in Ordered> = assoc_entity element2, %Greater [template = constants.%.5]
// CHECK:STDOUT:   %GreaterOrEquivalent: <function> = fn_decl @GreaterOrEquivalent [template] {
// CHECK:STDOUT:     %Self.ref.loc9_32: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_32.1: type = facet_type_access %Self.ref.loc9_32 [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_32.2: type = converted %Self.ref.loc9_32, %.loc9_32.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc9_26.1: Self = param self
// CHECK:STDOUT:     %self.loc9_26.2: Self = bind_name self, %self.loc9_26.1
// CHECK:STDOUT:     %Self.ref.loc9_45: Ordered = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_45.1: type = facet_type_access %Self.ref.loc9_45 [symbolic = %Self]
// CHECK:STDOUT:     %.loc9_45.2: type = converted %Self.ref.loc9_45, %.loc9_45.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc9_38.1: Self = param other
// CHECK:STDOUT:     %other.loc9_38.2: Self = bind_name other, %other.loc9_38.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_58: <associated <function> in Ordered> = assoc_entity element3, %GreaterOrEquivalent [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Less = %.loc6_43
// CHECK:STDOUT:   .LessOrEquivalent = %.loc7_55
// CHECK:STDOUT:   .Greater = %.loc8_46
// CHECK:STDOUT:   .GreaterOrEquivalent = %.loc9_58
// CHECK:STDOUT:   witness = (%Less, %LessOrEquivalent, %Greater, %GreaterOrEquivalent)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less[@Ordered.%self.loc6_11.2: Self](@Ordered.%other.loc6_23.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent[@Ordered.%self.loc7_23.2: Self](@Ordered.%other.loc7_35.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater[@Ordered.%self.loc8_14.2: Self](@Ordered.%other.loc8_26.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent[@Ordered.%self.loc9_26.2: Self](@Ordered.%other.loc9_38.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (@impl.%Less, @impl.%LessOrEquivalent, @impl.%Greater, @impl.%GreaterOrEquivalent) [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Ordered, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in Ordered> = assoc_entity element0, file.%import_ref.12 [template]
// CHECK:STDOUT:   %.8: <associated <function> in Ordered> = assoc_entity element1, file.%import_ref.14 [template]
// CHECK:STDOUT:   %.9: <associated <function> in Ordered> = assoc_entity element2, file.%import_ref.16 [template]
// CHECK:STDOUT:   %.10: <associated <function> in Ordered> = assoc_entity element3, file.%import_ref.18 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestLess = %TestLess
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual
// CHECK:STDOUT:     .TestGreater = %TestGreater
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Ordered> = import_ref ir1, inst+17, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <associated <function> in Ordered> = import_ref ir1, inst+31, used [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.5: <associated <function> in Ordered> = import_ref ir1, inst+59, used [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.6: <associated <function> in Ordered> = import_ref ir1, inst+45, used [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir1, inst+15, used [template = imports.%Less]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir1, inst+30, used [template = imports.%LessOrEquivalent]
// CHECK:STDOUT:   %import_ref.9: <function> = import_ref ir1, inst+44, used [template = imports.%Greater]
// CHECK:STDOUT:   %import_ref.10: <function> = import_ref ir1, inst+58, used [template = imports.%GreaterOrEquivalent]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Ordered.decl: invalid = interface_decl @Ordered [template = constants.%.2] {}
// CHECK:STDOUT:     %Ordered.ref: type = name_ref Ordered, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLess: <function> = fn_decl @TestLess [template] {
// CHECK:STDOUT:     %C.ref.loc15_16: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_13.1: C = param a
// CHECK:STDOUT:     @TestLess.%a: C = bind_name a, %a.loc15_13.1
// CHECK:STDOUT:     %C.ref.loc15_22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc15_19.1: C = param b
// CHECK:STDOUT:     @TestLess.%b: C = bind_name b, %b.loc15_19.1
// CHECK:STDOUT:     %return.var.loc15: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %TestLessEqual: <function> = fn_decl @TestLessEqual [template] {
// CHECK:STDOUT:     %C.ref.loc19_21: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc19_18.1: C = param a
// CHECK:STDOUT:     @TestLessEqual.%a: C = bind_name a, %a.loc19_18.1
// CHECK:STDOUT:     %C.ref.loc19_27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc19_24.1: C = param b
// CHECK:STDOUT:     @TestLessEqual.%b: C = bind_name b, %b.loc19_24.1
// CHECK:STDOUT:     %return.var.loc19: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.13: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.14 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %TestGreater: <function> = fn_decl @TestGreater [template] {
// CHECK:STDOUT:     %C.ref.loc23_19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc23_16.1: C = param a
// CHECK:STDOUT:     @TestGreater.%a: C = bind_name a, %a.loc23_16.1
// CHECK:STDOUT:     %C.ref.loc23_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc23_22.1: C = param b
// CHECK:STDOUT:     @TestGreater.%b: C = bind_name b, %b.loc23_22.1
// CHECK:STDOUT:     %return.var.loc23: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.15: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.16 = import_ref ir1, inst+44, unused
// CHECK:STDOUT:   %TestGreaterEqual: <function> = fn_decl @TestGreaterEqual [template] {
// CHECK:STDOUT:     %C.ref.loc27_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc27_21.1: C = param a
// CHECK:STDOUT:     @TestGreaterEqual.%a: C = bind_name a, %a.loc27_21.1
// CHECK:STDOUT:     %C.ref.loc27_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc27_27.1: C = param b
// CHECK:STDOUT:     @TestGreaterEqual.%b: C = bind_name b, %b.loc27_27.1
// CHECK:STDOUT:     %return.var.loc27: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.17: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.18 = import_ref ir1, inst+58, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .LessOrEquivalent = file.%import_ref.4
// CHECK:STDOUT:   .GreaterOrEquivalent = file.%import_ref.5
// CHECK:STDOUT:   .Greater = file.%import_ref.6
// CHECK:STDOUT:   witness = (file.%import_ref.7, file.%import_ref.8, file.%import_ref.9, file.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Ordered {
// CHECK:STDOUT:   %Less: <function> = fn_decl @Less.1 [template] {
// CHECK:STDOUT:     %C.ref.loc9_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_11.1: C = param self
// CHECK:STDOUT:     %self.loc9_11.2: C = bind_name self, %self.loc9_11.1
// CHECK:STDOUT:     %C.ref.loc9_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_20.1: C = param other
// CHECK:STDOUT:     %other.loc9_20.2: C = bind_name other, %other.loc9_20.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LessOrEquivalent: <function> = fn_decl @LessOrEquivalent.1 [template] {
// CHECK:STDOUT:     %C.ref.loc10_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc10_23.1: C = param self
// CHECK:STDOUT:     %self.loc10_23.2: C = bind_name self, %self.loc10_23.1
// CHECK:STDOUT:     %C.ref.loc10_39: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc10_32.1: C = param other
// CHECK:STDOUT:     %other.loc10_32.2: C = bind_name other, %other.loc10_32.1
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Greater: <function> = fn_decl @Greater.1 [template] {
// CHECK:STDOUT:     %C.ref.loc11_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc11_14.1: C = param self
// CHECK:STDOUT:     %self.loc11_14.2: C = bind_name self, %self.loc11_14.1
// CHECK:STDOUT:     %C.ref.loc11_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc11_23.1: C = param other
// CHECK:STDOUT:     %other.loc11_23.2: C = bind_name other, %other.loc11_23.1
// CHECK:STDOUT:     %return.var.loc11: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GreaterOrEquivalent: <function> = fn_decl @GreaterOrEquivalent.1 [template] {
// CHECK:STDOUT:     %C.ref.loc12_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc12_26.1: C = param self
// CHECK:STDOUT:     %self.loc12_26.2: C = bind_name self, %self.loc12_26.1
// CHECK:STDOUT:     %C.ref.loc12_42: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc12_35.1: C = param other
// CHECK:STDOUT:     %other.loc12_35.2: C = bind_name other, %other.loc12_35.1
// CHECK:STDOUT:     %return.var.loc12: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Less, %LessOrEquivalent, %Greater, %GreaterOrEquivalent) [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = %Less
// CHECK:STDOUT:   .LessOrEquivalent = %LessOrEquivalent
// CHECK:STDOUT:   .Greater = %Greater
// CHECK:STDOUT:   .GreaterOrEquivalent = %GreaterOrEquivalent
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less.1[@impl.%self.loc9_11.2: C](@impl.%other.loc9_20.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent.1[@impl.%self.loc10_23.2: C](@impl.%other.loc10_32.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater.1[@impl.%self.loc11_14.2: C](@impl.%other.loc11_23.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent.1[@impl.%self.loc12_26.2: C](@impl.%other.loc12_35.2: C) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent.2[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Less]
// CHECK:STDOUT:   %.loc16_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc16_12.2: init bool = call %.loc16_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_15: bool = value_of_initializer %.loc16_12.2
// CHECK:STDOUT:   %.loc16_12.3: bool = converted %.loc16_12.2, %.loc16_15
// CHECK:STDOUT:   return %.loc16_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element1 [template = @impl.%LessOrEquivalent]
// CHECK:STDOUT:   %.loc20_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc20_12.2: init bool = call %.loc20_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc20_16: bool = value_of_initializer %.loc20_12.2
// CHECK:STDOUT:   %.loc20_12.3: bool = converted %.loc20_12.2, %.loc20_16
// CHECK:STDOUT:   return %.loc20_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element2 [template = @impl.%Greater]
// CHECK:STDOUT:   %.loc24_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc24_12.2: init bool = call %.loc24_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc24_15: bool = value_of_initializer %.loc24_12.2
// CHECK:STDOUT:   %.loc24_12.3: bool = converted %.loc24_12.2, %.loc24_15
// CHECK:STDOUT:   return %.loc24_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element3 [template = @impl.%GreaterOrEquivalent]
// CHECK:STDOUT:   %.loc28_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc28_12.2: init bool = call %.loc28_12.1(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc28_16: bool = value_of_initializer %.loc28_12.2
// CHECK:STDOUT:   %.loc28_12.3: bool = converted %.loc28_12.2, %.loc28_16
// CHECK:STDOUT:   return %.loc28_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Ordered, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Ordered> = assoc_entity element0, file.%import_ref.11 [template]
// CHECK:STDOUT:   %.7: <associated <function> in Ordered> = assoc_entity element1, file.%import_ref.13 [template]
// CHECK:STDOUT:   %.8: <associated <function> in Ordered> = assoc_entity element2, file.%import_ref.15 [template]
// CHECK:STDOUT:   %.9: <associated <function> in Ordered> = assoc_entity element3, file.%import_ref.17 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestLess = %TestLess
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual
// CHECK:STDOUT:     .TestGreater = %TestGreater
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestLess: <function> = fn_decl @TestLess [template] {
// CHECK:STDOUT:     %D.ref.loc8_16: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc8_13.1: D = param a
// CHECK:STDOUT:     @TestLess.%a: D = bind_name a, %a.loc8_13.1
// CHECK:STDOUT:     %D.ref.loc8_22: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc8_19.1: D = param b
// CHECK:STDOUT:     @TestLess.%b: D = bind_name b, %b.loc8_19.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Ordered> = import_ref ir1, inst+17, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <associated <function> in Ordered> = import_ref ir1, inst+31, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.5: <associated <function> in Ordered> = import_ref ir1, inst+59, used [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.6: <associated <function> in Ordered> = import_ref ir1, inst+45, used [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %import_ref.8 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %import_ref.9 = import_ref ir1, inst+44, unused
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+58, unused
// CHECK:STDOUT:   %import_ref.11 = import_ref ir1, inst+15, unused
// CHECK:STDOUT:   %TestLessEqual: <function> = fn_decl @TestLessEqual [template] {
// CHECK:STDOUT:     %D.ref.loc16_21: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc16_18.1: D = param a
// CHECK:STDOUT:     @TestLessEqual.%a: D = bind_name a, %a.loc16_18.1
// CHECK:STDOUT:     %D.ref.loc16_27: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc16_24.1: D = param b
// CHECK:STDOUT:     @TestLessEqual.%b: D = bind_name b, %b.loc16_24.1
// CHECK:STDOUT:     %return.var.loc16: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.12: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.13 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %TestGreater: <function> = fn_decl @TestGreater [template] {
// CHECK:STDOUT:     %D.ref.loc24_19: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc24_16.1: D = param a
// CHECK:STDOUT:     @TestGreater.%a: D = bind_name a, %a.loc24_16.1
// CHECK:STDOUT:     %D.ref.loc24_25: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc24_22.1: D = param b
// CHECK:STDOUT:     @TestGreater.%b: D = bind_name b, %b.loc24_22.1
// CHECK:STDOUT:     %return.var.loc24: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.14: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.15 = import_ref ir1, inst+44, unused
// CHECK:STDOUT:   %TestGreaterEqual: <function> = fn_decl @TestGreaterEqual [template] {
// CHECK:STDOUT:     %D.ref.loc32_24: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc32_21.1: D = param a
// CHECK:STDOUT:     @TestGreaterEqual.%a: D = bind_name a, %a.loc32_21.1
// CHECK:STDOUT:     %D.ref.loc32_30: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc32_27.1: D = param b
// CHECK:STDOUT:     @TestGreaterEqual.%b: D = bind_name b, %b.loc32_27.1
// CHECK:STDOUT:     %return.var.loc32: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.16: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.17 = import_ref ir1, inst+58, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .LessOrEquivalent = file.%import_ref.4
// CHECK:STDOUT:   .GreaterOrEquivalent = file.%import_ref.5
// CHECK:STDOUT:   .Greater = file.%import_ref.6
// CHECK:STDOUT:   witness = (file.%import_ref.7, file.%import_ref.8, file.%import_ref.9, file.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLess(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %Ordered.decl: invalid = interface_decl @Ordered [template = constants.%.4] {}
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
