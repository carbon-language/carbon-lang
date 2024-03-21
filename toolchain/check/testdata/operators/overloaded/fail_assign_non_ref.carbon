// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Inc {
  fn Op[addr self: Self*]();
}
interface AddAssign {
  fn Op[addr self: Self*](other: Self);
}

// --- fail_assign_non_ref.carbon

package User api;

import Core;

class C {};

impl C as Core.Inc {
  fn Op[addr self: C*]();
}
impl C as Core.AddAssign {
  fn Op[addr self: C*](other: C);
}

fn TestIncNonRef(a: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+7]]:3: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   ++a;
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-10]]:14: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Op[addr self: C*]();
  // CHECK:STDERR:              ^~~~
  // CHECK:STDERR:
  ++a;
}

fn TestAddAssignNonRef(a: C, b: C) {
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE+6]]:5: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR: fail_assign_non_ref.carbon:[[@LINE-18]]:14: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Op[addr self: C*](other: C);
  // CHECK:STDERR:              ^~~~
  a += b;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Inc [template]
// CHECK:STDOUT:   %.2: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @Inc, <function> [template]
// CHECK:STDOUT:   %.4: <associated <function> in Inc> = assoc_entity element0, @Inc.%Op [template]
// CHECK:STDOUT:   %.5: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %.6: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @AddAssign, <function> [template]
// CHECK:STDOUT:   %.8: <associated <function> in AddAssign> = assoc_entity element0, @AddAssign.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Inc = %Inc.decl
// CHECK:STDOUT:     .AddAssign = %AddAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inc.decl: type = interface_decl @Inc [template = constants.%.1] {}
// CHECK:STDOUT:   %AddAssign.decl: type = interface_decl @AddAssign [template = constants.%.5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT:   %Self: Inc = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %Self.ref: Inc = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_24.1: type = facet_type_access %Self.ref [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_20: type = converted %Self.ref, %.loc5_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_24.2: type = ptr_type Self [symbolic = constants.%.2]
// CHECK:STDOUT:     %self.loc5_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc5_14.3: Self* = bind_name self, %self.loc5_14.1
// CHECK:STDOUT:     %.loc5_9: Self* = addr_pattern %self.loc5_14.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_28: <associated <function> in Inc> = assoc_entity element0, %Op [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_28
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT:   %Self: AddAssign = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_20: AddAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.1: type = facet_type_access %Self.ref.loc8_20 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_20: type = converted %Self.ref.loc8_20, %.loc8_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.2: type = ptr_type Self [symbolic = constants.%.6]
// CHECK:STDOUT:     %self.loc8_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc8_14.3: Self* = bind_name self, %self.loc8_14.1
// CHECK:STDOUT:     %.loc8_9: Self* = addr_pattern %self.loc8_14.3
// CHECK:STDOUT:     %Self.ref.loc8_34: AddAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.1: type = facet_type_access %Self.ref.loc8_34 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %Self.ref.loc8_34, %.loc8_34.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_27.1: Self = param other
// CHECK:STDOUT:     %other.loc8_27.2: Self = bind_name other, %other.loc8_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_39: <associated <function> in AddAssign> = assoc_entity element0, %Op [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8_39
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[addr @Inc.%self.loc5_14.3: Self*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr @AddAssign.%self.loc8_14.3: Self*](@AddAssign.%other.loc8_27.2: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_assign_non_ref.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Inc [template]
// CHECK:STDOUT:   %.3: type = ptr_type C [template]
// CHECK:STDOUT:   %.4: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (@impl.1.%Op) [template]
// CHECK:STDOUT:   %.6: type = interface_type @AddAssign [template]
// CHECK:STDOUT:   %.7: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (@impl.2.%Op) [template]
// CHECK:STDOUT:   %.9: type = tuple_type () [template]
// CHECK:STDOUT:   %.10: type = ptr_type {} [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type @Inc, <function> [template]
// CHECK:STDOUT:   %.12: <associated <function> in Inc> = assoc_entity element0, file.%import_ref.10 [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type @AddAssign, <function> [template]
// CHECK:STDOUT:   %.14: <associated <function> in AddAssign> = assoc_entity element0, file.%import_ref.12 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestIncNonRef = %TestIncNonRef
// CHECK:STDOUT:     .TestAddAssignNonRef = %TestAddAssignNonRef
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Inc> = import_ref ir1, inst+14, used [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+12, used [template = imports.%Op.1]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc8: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Inc.decl: invalid = interface_decl @Inc [template = constants.%.2] {}
// CHECK:STDOUT:     %Inc.ref: type = name_ref Inc, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir1, inst+16, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.6: <associated <function> in AddAssign> = import_ref ir1, inst+34, used [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir1, inst+32, used [template = imports.%Op.2]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc11: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc11: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %AddAssign.decl: invalid = interface_decl @AddAssign [template = constants.%.6] {}
// CHECK:STDOUT:     %AddAssign.ref: type = name_ref AddAssign, %import_ref.5 [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestIncNonRef: <function> = fn_decl @TestIncNonRef [template] {
// CHECK:STDOUT:     %C.ref.loc15: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc15_18.1: C = param a
// CHECK:STDOUT:     @TestIncNonRef.%a: C = bind_name a, %a.loc15_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+12, unused
// CHECK:STDOUT:   %TestAddAssignNonRef: <function> = fn_decl @TestAddAssignNonRef [template] {
// CHECK:STDOUT:     %C.ref.loc26_27: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc26_24.1: C = param a
// CHECK:STDOUT:     @TestAddAssignNonRef.%a: C = bind_name a, %a.loc26_24.1
// CHECK:STDOUT:     %C.ref.loc26_33: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc26_30.1: C = param b
// CHECK:STDOUT:     @TestAddAssignNonRef.%b: C = bind_name b, %b.loc26_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir1, inst+16, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir1, inst+32, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inc {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AddAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT:   witness = (file.%import_ref.8)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as Inc {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc9_21: type = ptr_type C [template = constants.%.3]
// CHECK:STDOUT:     %self.loc9_14.1: C* = param self
// CHECK:STDOUT:     %self.loc9_14.3: C* = bind_name self, %self.loc9_14.1
// CHECK:STDOUT:     %.loc9_9: C* = addr_pattern %self.loc9_14.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: C as AddAssign {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.3 [template] {
// CHECK:STDOUT:     %C.ref.loc12_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc12_21: type = ptr_type C [template = constants.%.3]
// CHECK:STDOUT:     %self.loc12_14.1: C* = param self
// CHECK:STDOUT:     %self.loc12_14.3: C* = bind_name self, %self.loc12_14.1
// CHECK:STDOUT:     %.loc12_9: C* = addr_pattern %self.loc12_14.3
// CHECK:STDOUT:     %C.ref.loc12_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc12_24.1: C = param other
// CHECK:STDOUT:     %other.loc12_24.2: C = bind_name other, %other.loc12_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[addr @impl.1.%self.loc9_14.3: C*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr %self: Self*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr @impl.2.%self.loc12_14.3: C*](@impl.2.%other.loc12_24.2: C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self: Self*](%other: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIncNonRef(%a: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.1.%.1, element0 [template = @impl.1.%Op]
// CHECK:STDOUT:   %.loc23_3.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc23_3.2: init () = call %.loc23_3.1(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAddAssignNonRef(%a: C, %b: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.2.%.1, element0 [template = @impl.2.%Op]
// CHECK:STDOUT:   %.loc33_5.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc33_5.2: init () = call %.loc33_5.1(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
