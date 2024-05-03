// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

import library "prelude";

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
// CHECK:STDOUT:   %Self: Ordered = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Less: type = fn_type @Less [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Less = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Ordered, Less [template]
// CHECK:STDOUT:   %.4: <associated Less in Ordered> = assoc_entity element0, @Ordered.%Less.decl [template]
// CHECK:STDOUT:   %LessOrEquivalent: type = fn_type @LessOrEquivalent [template]
// CHECK:STDOUT:   %struct.2: LessOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Ordered, LessOrEquivalent [template]
// CHECK:STDOUT:   %.6: <associated LessOrEquivalent in Ordered> = assoc_entity element1, @Ordered.%LessOrEquivalent.decl [template]
// CHECK:STDOUT:   %Greater: type = fn_type @Greater [template]
// CHECK:STDOUT:   %struct.3: Greater = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @Ordered, Greater [template]
// CHECK:STDOUT:   %.8: <associated Greater in Ordered> = assoc_entity element2, @Ordered.%Greater.decl [template]
// CHECK:STDOUT:   %GreaterOrEquivalent: type = fn_type @GreaterOrEquivalent [template]
// CHECK:STDOUT:   %struct.4: GreaterOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type @Ordered, GreaterOrEquivalent [template]
// CHECK:STDOUT:   %.10: <associated GreaterOrEquivalent in Ordered> = assoc_entity element3, @Ordered.%GreaterOrEquivalent.decl [template]
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
// CHECK:STDOUT:   %Self: Ordered = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Less.decl: Less = fn_decl @Less [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc8_17: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_17.1: type = facet_type_access %Self.ref.loc8_17 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_17.2: type = converted %Self.ref.loc8_17, %.loc8_17.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc8_11.1: Self = param self
// CHECK:STDOUT:     %self.loc8_11.2: Self = bind_name self, %self.loc8_11.1
// CHECK:STDOUT:     %Self.ref.loc8_30: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_30.1: type = facet_type_access %Self.ref.loc8_30 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc8_30.2: type = converted %Self.ref.loc8_30, %.loc8_30.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %other.loc8_23.1: Self = param other
// CHECK:STDOUT:     %other.loc8_23.2: Self = bind_name other, %other.loc8_23.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_43: <associated Less in Ordered> = assoc_entity element0, %Less.decl [template = constants.%.4]
// CHECK:STDOUT:   %LessOrEquivalent.decl: LessOrEquivalent = fn_decl @LessOrEquivalent [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc9_29: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_29.1: type = facet_type_access %Self.ref.loc9_29 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_29.2: type = converted %Self.ref.loc9_29, %.loc9_29.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc9_23.1: Self = param self
// CHECK:STDOUT:     %self.loc9_23.2: Self = bind_name self, %self.loc9_23.1
// CHECK:STDOUT:     %Self.ref.loc9_42: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_42.1: type = facet_type_access %Self.ref.loc9_42 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc9_42.2: type = converted %Self.ref.loc9_42, %.loc9_42.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %other.loc9_35.1: Self = param other
// CHECK:STDOUT:     %other.loc9_35.2: Self = bind_name other, %other.loc9_35.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_55: <associated LessOrEquivalent in Ordered> = assoc_entity element1, %LessOrEquivalent.decl [template = constants.%.6]
// CHECK:STDOUT:   %Greater.decl: Greater = fn_decl @Greater [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc10_20: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc10_20.1: type = facet_type_access %Self.ref.loc10_20 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc10_20.2: type = converted %Self.ref.loc10_20, %.loc10_20.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc10_14.1: Self = param self
// CHECK:STDOUT:     %self.loc10_14.2: Self = bind_name self, %self.loc10_14.1
// CHECK:STDOUT:     %Self.ref.loc10_33: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc10_33.1: type = facet_type_access %Self.ref.loc10_33 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc10_33.2: type = converted %Self.ref.loc10_33, %.loc10_33.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %other.loc10_26.1: Self = param other
// CHECK:STDOUT:     %other.loc10_26.2: Self = bind_name other, %other.loc10_26.1
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_46: <associated Greater in Ordered> = assoc_entity element2, %Greater.decl [template = constants.%.8]
// CHECK:STDOUT:   %GreaterOrEquivalent.decl: GreaterOrEquivalent = fn_decl @GreaterOrEquivalent [template = constants.%struct.4] {
// CHECK:STDOUT:     %Self.ref.loc11_32: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc11_32.1: type = facet_type_access %Self.ref.loc11_32 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc11_32.2: type = converted %Self.ref.loc11_32, %.loc11_32.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc11_26.1: Self = param self
// CHECK:STDOUT:     %self.loc11_26.2: Self = bind_name self, %self.loc11_26.1
// CHECK:STDOUT:     %Self.ref.loc11_45: Ordered = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc11_45.1: type = facet_type_access %Self.ref.loc11_45 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc11_45.2: type = converted %Self.ref.loc11_45, %.loc11_45.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %other.loc11_38.1: Self = param other
// CHECK:STDOUT:     %other.loc11_38.2: Self = bind_name other, %other.loc11_38.1
// CHECK:STDOUT:     %return.var.loc11: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_58: <associated GreaterOrEquivalent in Ordered> = assoc_entity element3, %GreaterOrEquivalent.decl [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Less = %.loc8_43
// CHECK:STDOUT:   .LessOrEquivalent = %.loc9_55
// CHECK:STDOUT:   .Greater = %.loc10_46
// CHECK:STDOUT:   .GreaterOrEquivalent = %.loc11_58
// CHECK:STDOUT:   witness = (%Less.decl, %LessOrEquivalent.decl, %Greater.decl, %GreaterOrEquivalent.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less[@Ordered.%self.loc8_11.2: Self](@Ordered.%other.loc8_23.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent[@Ordered.%self.loc9_23.2: Self](@Ordered.%other.loc9_35.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater[@Ordered.%self.loc10_14.2: Self](@Ordered.%other.loc10_26.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent[@Ordered.%self.loc11_26.2: Self](@Ordered.%other.loc11_38.2: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %Self: Ordered = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Less.1: type = fn_type @Less.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Less = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent.1: type = fn_type @LessOrEquivalent.1 [template]
// CHECK:STDOUT:   %struct.2: LessOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %Greater.1: type = fn_type @Greater.1 [template]
// CHECK:STDOUT:   %struct.3: Greater = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.1: type = fn_type @GreaterOrEquivalent.1 [template]
// CHECK:STDOUT:   %struct.4: GreaterOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %Less.2: type = fn_type @Less.2 [template]
// CHECK:STDOUT:   %struct.5: Less = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent.2: type = fn_type @LessOrEquivalent.2 [template]
// CHECK:STDOUT:   %struct.6: LessOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %Greater.2: type = fn_type @Greater.2 [template]
// CHECK:STDOUT:   %struct.7: Greater = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent.2: type = fn_type @GreaterOrEquivalent.2 [template]
// CHECK:STDOUT:   %struct.8: GreaterOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%struct.1, %struct.2, %struct.3, %struct.4) [template]
// CHECK:STDOUT:   %TestLess: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %struct.9: TestLess = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Ordered, Less [template]
// CHECK:STDOUT:   %.7: <associated Less in Ordered> = assoc_entity element0, file.%import_ref.12 [template]
// CHECK:STDOUT:   %TestLessEqual: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %struct.10: TestLessEqual = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type @Ordered, LessOrEquivalent [template]
// CHECK:STDOUT:   %.9: <associated LessOrEquivalent in Ordered> = assoc_entity element1, file.%import_ref.14 [template]
// CHECK:STDOUT:   %TestGreater: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %struct.11: TestGreater = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type @Ordered, Greater [template]
// CHECK:STDOUT:   %.11: <associated Greater in Ordered> = assoc_entity element2, file.%import_ref.16 [template]
// CHECK:STDOUT:   %TestGreaterEqual: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %struct.12: TestGreaterEqual = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type @Ordered, GreaterOrEquivalent [template]
// CHECK:STDOUT:   %.13: <associated GreaterOrEquivalent in Ordered> = assoc_entity element3, file.%import_ref.18 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated GreaterOrEquivalent in Ordered> = import_ref ir2, inst+72, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: <associated Greater in Ordered> = import_ref ir2, inst+55, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.5: <associated Less in Ordered> = import_ref ir2, inst+21, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.6: <associated LessOrEquivalent in Ordered> = import_ref ir2, inst+38, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.7: Less = import_ref ir2, inst+16, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %import_ref.8: LessOrEquivalent = import_ref ir2, inst+34, loaded [template = constants.%struct.6]
// CHECK:STDOUT:   %import_ref.9: Greater = import_ref ir2, inst+51, loaded [template = constants.%struct.7]
// CHECK:STDOUT:   %import_ref.10: GreaterOrEquivalent = import_ref ir2, inst+68, loaded [template = constants.%struct.8]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Ordered.ref: type = name_ref Ordered, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLess.decl: TestLess = fn_decl @TestLess [template = constants.%struct.9] {
// CHECK:STDOUT:     %C.ref.loc15_16: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_13.1: C = param a
// CHECK:STDOUT:     @TestLess.%a: C = bind_name a, %a.loc15_13.1
// CHECK:STDOUT:     %C.ref.loc15_22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc15_19.1: C = param b
// CHECK:STDOUT:     @TestLess.%b: C = bind_name b, %b.loc15_19.1
// CHECK:STDOUT:     @TestLess.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir2, inst+16, unloaded
// CHECK:STDOUT:   %TestLessEqual.decl: TestLessEqual = fn_decl @TestLessEqual [template = constants.%struct.10] {
// CHECK:STDOUT:     %C.ref.loc19_21: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc19_18.1: C = param a
// CHECK:STDOUT:     @TestLessEqual.%a: C = bind_name a, %a.loc19_18.1
// CHECK:STDOUT:     %C.ref.loc19_27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc19_24.1: C = param b
// CHECK:STDOUT:     @TestLessEqual.%b: C = bind_name b, %b.loc19_24.1
// CHECK:STDOUT:     @TestLessEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.13: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.14 = import_ref ir2, inst+34, unloaded
// CHECK:STDOUT:   %TestGreater.decl: TestGreater = fn_decl @TestGreater [template = constants.%struct.11] {
// CHECK:STDOUT:     %C.ref.loc23_19: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc23_16.1: C = param a
// CHECK:STDOUT:     @TestGreater.%a: C = bind_name a, %a.loc23_16.1
// CHECK:STDOUT:     %C.ref.loc23_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc23_22.1: C = param b
// CHECK:STDOUT:     @TestGreater.%b: C = bind_name b, %b.loc23_22.1
// CHECK:STDOUT:     @TestGreater.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.15: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.16 = import_ref ir2, inst+51, unloaded
// CHECK:STDOUT:   %TestGreaterEqual.decl: TestGreaterEqual = fn_decl @TestGreaterEqual [template = constants.%struct.12] {
// CHECK:STDOUT:     %C.ref.loc27_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc27_21.1: C = param a
// CHECK:STDOUT:     @TestGreaterEqual.%a: C = bind_name a, %a.loc27_21.1
// CHECK:STDOUT:     %C.ref.loc27_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc27_27.1: C = param b
// CHECK:STDOUT:     @TestGreaterEqual.%b: C = bind_name b, %b.loc27_27.1
// CHECK:STDOUT:     @TestGreaterEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.17: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.18 = import_ref ir2, inst+68, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .GreaterOrEquivalent = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .Greater = file.%import_ref.4
// CHECK:STDOUT:   .Less = file.%import_ref.5
// CHECK:STDOUT:   .LessOrEquivalent = file.%import_ref.6
// CHECK:STDOUT:   witness = (file.%import_ref.7, file.%import_ref.8, file.%import_ref.9, file.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Ordered {
// CHECK:STDOUT:   %Less.decl: Less = fn_decl @Less.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc9_17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_11.1: C = param self
// CHECK:STDOUT:     %self.loc9_11.2: C = bind_name self, %self.loc9_11.1
// CHECK:STDOUT:     %C.ref.loc9_27: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_20.1: C = param other
// CHECK:STDOUT:     %other.loc9_20.2: C = bind_name other, %other.loc9_20.1
// CHECK:STDOUT:     %return.var.loc9: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LessOrEquivalent.decl: LessOrEquivalent = fn_decl @LessOrEquivalent.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref.loc10_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc10_23.1: C = param self
// CHECK:STDOUT:     %self.loc10_23.2: C = bind_name self, %self.loc10_23.1
// CHECK:STDOUT:     %C.ref.loc10_39: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc10_32.1: C = param other
// CHECK:STDOUT:     %other.loc10_32.2: C = bind_name other, %other.loc10_32.1
// CHECK:STDOUT:     %return.var.loc10: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Greater.decl: Greater = fn_decl @Greater.1 [template = constants.%struct.3] {
// CHECK:STDOUT:     %C.ref.loc11_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc11_14.1: C = param self
// CHECK:STDOUT:     %self.loc11_14.2: C = bind_name self, %self.loc11_14.1
// CHECK:STDOUT:     %C.ref.loc11_30: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc11_23.1: C = param other
// CHECK:STDOUT:     %other.loc11_23.2: C = bind_name other, %other.loc11_23.1
// CHECK:STDOUT:     %return.var.loc11: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GreaterOrEquivalent.decl: GreaterOrEquivalent = fn_decl @GreaterOrEquivalent.1 [template = constants.%struct.4] {
// CHECK:STDOUT:     %C.ref.loc12_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc12_26.1: C = param self
// CHECK:STDOUT:     %self.loc12_26.2: C = bind_name self, %self.loc12_26.1
// CHECK:STDOUT:     %C.ref.loc12_42: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc12_35.1: C = param other
// CHECK:STDOUT:     %other.loc12_35.2: C = bind_name other, %other.loc12_35.1
// CHECK:STDOUT:     %return.var.loc12: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Less.decl, %LessOrEquivalent.decl, %Greater.decl, %GreaterOrEquivalent.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Less = %Less.decl
// CHECK:STDOUT:   .LessOrEquivalent = %LessOrEquivalent.decl
// CHECK:STDOUT:   .Greater = %Greater.decl
// CHECK:STDOUT:   .GreaterOrEquivalent = %GreaterOrEquivalent.decl
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
// CHECK:STDOUT:   %.1: Less = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc16_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Less.call: init bool = call %.loc16_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_15.1: bool = value_of_initializer %Less.call
// CHECK:STDOUT:   %.loc16_15.2: bool = converted %Less.call, %.loc16_15.1
// CHECK:STDOUT:   return %.loc16_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: LessOrEquivalent = interface_witness_access @impl.%.1, element1 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc20_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %LessOrEquivalent.call: init bool = call %.loc20_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc20_16.1: bool = value_of_initializer %LessOrEquivalent.call
// CHECK:STDOUT:   %.loc20_16.2: bool = converted %LessOrEquivalent.call, %.loc20_16.1
// CHECK:STDOUT:   return %.loc20_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: Greater = interface_witness_access @impl.%.1, element2 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc24_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %Greater.call: init bool = call %.loc24_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc24_15.1: bool = value_of_initializer %Greater.call
// CHECK:STDOUT:   %.loc24_15.2: bool = converted %Greater.call, %.loc24_15.1
// CHECK:STDOUT:   return %.loc24_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: C, %b: C) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: GreaterOrEquivalent = interface_witness_access @impl.%.1, element3 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc28_12: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %GreaterOrEquivalent.call: init bool = call %.loc28_12(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc28_16.1: bool = value_of_initializer %GreaterOrEquivalent.call
// CHECK:STDOUT:   %.loc28_16.2: bool = converted %GreaterOrEquivalent.call, %.loc28_16.1
// CHECK:STDOUT:   return %.loc28_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %TestLess: type = fn_type @TestLess [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: TestLess = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Ordered [template]
// CHECK:STDOUT:   %Self: Ordered = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Less: type = fn_type @Less [template]
// CHECK:STDOUT:   %struct.2: Less = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Ordered, Less [template]
// CHECK:STDOUT:   %.6: <associated Less in Ordered> = assoc_entity element0, file.%import_ref.11 [template]
// CHECK:STDOUT:   %TestLessEqual: type = fn_type @TestLessEqual [template]
// CHECK:STDOUT:   %struct.3: TestLessEqual = struct_value () [template]
// CHECK:STDOUT:   %LessOrEquivalent: type = fn_type @LessOrEquivalent [template]
// CHECK:STDOUT:   %struct.4: LessOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @Ordered, LessOrEquivalent [template]
// CHECK:STDOUT:   %.8: <associated LessOrEquivalent in Ordered> = assoc_entity element1, file.%import_ref.13 [template]
// CHECK:STDOUT:   %TestGreater: type = fn_type @TestGreater [template]
// CHECK:STDOUT:   %struct.5: TestGreater = struct_value () [template]
// CHECK:STDOUT:   %Greater: type = fn_type @Greater [template]
// CHECK:STDOUT:   %struct.6: Greater = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type @Ordered, Greater [template]
// CHECK:STDOUT:   %.10: <associated Greater in Ordered> = assoc_entity element2, file.%import_ref.15 [template]
// CHECK:STDOUT:   %TestGreaterEqual: type = fn_type @TestGreaterEqual [template]
// CHECK:STDOUT:   %struct.7: TestGreaterEqual = struct_value () [template]
// CHECK:STDOUT:   %GreaterOrEquivalent: type = fn_type @GreaterOrEquivalent [template]
// CHECK:STDOUT:   %struct.8: GreaterOrEquivalent = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type @Ordered, GreaterOrEquivalent [template]
// CHECK:STDOUT:   %.12: <associated GreaterOrEquivalent in Ordered> = assoc_entity element3, file.%import_ref.17 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestLess = %TestLess.decl
// CHECK:STDOUT:     .TestLessEqual = %TestLessEqual.decl
// CHECK:STDOUT:     .TestGreater = %TestGreater.decl
// CHECK:STDOUT:     .TestGreaterEqual = %TestGreaterEqual.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestLess.decl: TestLess = fn_decl @TestLess [template = constants.%struct.1] {
// CHECK:STDOUT:     %D.ref.loc8_16: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc8_13.1: D = param a
// CHECK:STDOUT:     @TestLess.%a: D = bind_name a, %a.loc8_13.1
// CHECK:STDOUT:     %D.ref.loc8_22: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc8_19.1: D = param b
// CHECK:STDOUT:     @TestLess.%b: D = bind_name b, %b.loc8_19.1
// CHECK:STDOUT:     @TestLess.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.2: <associated GreaterOrEquivalent in Ordered> = import_ref ir2, inst+72, loaded [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: <associated Greater in Ordered> = import_ref ir2, inst+55, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.5: <associated Less in Ordered> = import_ref ir2, inst+21, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.6: <associated LessOrEquivalent in Ordered> = import_ref ir2, inst+38, loaded [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+16, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+34, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+51, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+68, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+16, unloaded
// CHECK:STDOUT:   %TestLessEqual.decl: TestLessEqual = fn_decl @TestLessEqual [template = constants.%struct.3] {
// CHECK:STDOUT:     %D.ref.loc16_21: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc16_18.1: D = param a
// CHECK:STDOUT:     @TestLessEqual.%a: D = bind_name a, %a.loc16_18.1
// CHECK:STDOUT:     %D.ref.loc16_27: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc16_24.1: D = param b
// CHECK:STDOUT:     @TestLessEqual.%b: D = bind_name b, %b.loc16_24.1
// CHECK:STDOUT:     @TestLessEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.12: type = import_ref ir2, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.13 = import_ref ir2, inst+34, unloaded
// CHECK:STDOUT:   %TestGreater.decl: TestGreater = fn_decl @TestGreater [template = constants.%struct.5] {
// CHECK:STDOUT:     %D.ref.loc24_19: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc24_16.1: D = param a
// CHECK:STDOUT:     @TestGreater.%a: D = bind_name a, %a.loc24_16.1
// CHECK:STDOUT:     %D.ref.loc24_25: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc24_22.1: D = param b
// CHECK:STDOUT:     @TestGreater.%b: D = bind_name b, %b.loc24_22.1
// CHECK:STDOUT:     @TestGreater.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.14: type = import_ref ir2, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.15 = import_ref ir2, inst+51, unloaded
// CHECK:STDOUT:   %TestGreaterEqual.decl: TestGreaterEqual = fn_decl @TestGreaterEqual [template = constants.%struct.7] {
// CHECK:STDOUT:     %D.ref.loc32_24: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc32_21.1: D = param a
// CHECK:STDOUT:     @TestGreaterEqual.%a: D = bind_name a, %a.loc32_21.1
// CHECK:STDOUT:     %D.ref.loc32_30: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc32_27.1: D = param b
// CHECK:STDOUT:     @TestGreaterEqual.%b: D = bind_name b, %b.loc32_27.1
// CHECK:STDOUT:     @TestGreaterEqual.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.16: type = import_ref ir2, inst+1, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.17 = import_ref ir2, inst+68, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Ordered {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .GreaterOrEquivalent = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   .Greater = file.%import_ref.4
// CHECK:STDOUT:   .Less = file.%import_ref.5
// CHECK:STDOUT:   .LessOrEquivalent = file.%import_ref.6
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
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLessEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessOrEquivalent[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreater(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGreaterEqual(%a: D, %b: D) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterOrEquivalent[%self: Self](%other: Self) -> bool;
// CHECK:STDOUT:
