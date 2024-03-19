// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from binary_op.carbon.in. Run make_tests.sh to regenerate.

// --- prelude.carbon

package Core api;

interface Sub {
  fn Op[self: Self](other: Self) -> Self;
}
interface SubAssign {
  fn Op[addr self: Self*](other: Self);
}

// --- user.carbon

package User api;

import Core;

class C {};

impl C as Core.Sub {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.SubAssign {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  return a - b;
}

fn TestAssign(a: C*, b: C) {
  *a -= b;
}

// --- fail_no_impl.carbon

package FailNoImpl api;

import Core;

class D {};

fn TestOp(a: D, b: D) -> D {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Sub in type D that does not implement that interface.
  // CHECK:STDERR:   return a - b;
  // CHECK:STDERR:          ^~~~~
  return a - b;
}

fn TestAssign(a: D*, b: D) {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:3: ERROR: Cannot access member of interface SubAssign in type D that does not implement that interface.
  // CHECK:STDERR:   *a -= b;
  // CHECK:STDERR:   ^~~~~~~
  *a -= b;
}

// --- fail_no_impl_for_args.carbon

package FailNoImplForArgs api;

import Core;

class C {};

impl C as Core.Sub {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}

class D {};

fn TestRhsBad(a: C, b: D) -> C {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+6]]:10: ERROR: Cannot implicitly convert from `D` to `C`.
  // CHECK:STDERR:   return a - b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE-11]]:3: Initializing parameter 1 of function declared here.
  // CHECK:STDERR:   fn Op[self: C](other: C) -> C {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return a - b;
}

fn TestLhsBad(a: D, b: C) -> C {
  // CHECK:STDERR: fail_no_impl_for_args.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface Sub in type D that does not implement that interface.
  // CHECK:STDERR:   return a - b;
  // CHECK:STDERR:          ^~~~~
  return a - b;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Sub [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Sub, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Sub> = assoc_entity element0, @Sub.%Op [template]
// CHECK:STDOUT:   %.4: type = interface_type @SubAssign [template]
// CHECK:STDOUT:   %.5: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @SubAssign, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in SubAssign> = assoc_entity element0, @SubAssign.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .SubAssign = %SubAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: type = interface_decl @Sub [template = constants.%.1] {}
// CHECK:STDOUT:   %SubAssign.decl: type = interface_decl @SubAssign [template = constants.%.4] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Sub {
// CHECK:STDOUT:   %Self: Sub = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc5_15: Sub = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_28: Sub = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_type_access %Self.ref.loc5_28 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc5_21.1: Self = param other
// CHECK:STDOUT:     %other.loc5_21.2: Self = bind_name other, %other.loc5_21.1
// CHECK:STDOUT:     %Self.ref.loc5_37: Sub = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_type_access %Self.ref.loc5_37 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_41: <associated <function> in Sub> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_41
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SubAssign {
// CHECK:STDOUT:   %Self: SubAssign = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_20: SubAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.1: type = facet_type_access %Self.ref.loc8_20 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_20: type = converted %Self.ref.loc8_20, %.loc8_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.2: type = ptr_type Self [symbolic = constants.%.5]
// CHECK:STDOUT:     %self.loc8_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc8_14.3: Self* = bind_name self, %self.loc8_14.1
// CHECK:STDOUT:     %.loc8_9: Self* = addr_pattern %self.loc8_14.3
// CHECK:STDOUT:     %Self.ref.loc8_34: SubAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.1: type = facet_type_access %Self.ref.loc8_34 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %Self.ref.loc8_34, %.loc8_34.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_27.1: Self = param other
// CHECK:STDOUT:     %other.loc8_27.2: Self = bind_name other, %other.loc8_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_39: <associated <function> in SubAssign> = assoc_entity element0, %Op [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8_39
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@Sub.%self.loc5_9.2: Self](@Sub.%other.loc5_21.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr @SubAssign.%self.loc8_14.3: Self*](@SubAssign.%other.loc8_27.2: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Sub [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: C = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (@impl.1.%Op) [template]
// CHECK:STDOUT:   %.7: type = interface_type @SubAssign [template]
// CHECK:STDOUT:   %.8: type = ptr_type C [template]
// CHECK:STDOUT:   %.9: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (@impl.2.%Op) [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type @Sub, <function> [template]
// CHECK:STDOUT:   %.12: <associated <function> in Sub> = assoc_entity element0, @TestOp.%import_ref.2 [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type @SubAssign, <function> [template]
// CHECK:STDOUT:   %.14: <associated <function> in SubAssign> = assoc_entity element0, @TestAssign.%import_ref.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp
// CHECK:STDOUT:     .TestAssign = %TestAssign
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc8: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:     %Sub.decl: invalid = interface_decl @Sub [template = constants.%.2] {}
// CHECK:STDOUT:     %Sub.ref: type = name_ref Sub, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc13: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %import_ref.2: type = import_ref ir1, inst+22, used [template = constants.%.7]
// CHECK:STDOUT:     %SubAssign.decl: invalid = interface_decl @SubAssign [template = constants.%.7] {}
// CHECK:STDOUT:     %SubAssign.ref: type = name_ref SubAssign, %import_ref.2 [template = constants.%.7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp: <function> = fn_decl @TestOp [template] {
// CHECK:STDOUT:     %C.ref.loc17_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc17_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc17_11.1
// CHECK:STDOUT:     %C.ref.loc17_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc17_17.1: C = param b
// CHECK:STDOUT:     @TestOp.%b: C = bind_name b, %b.loc17_17.1
// CHECK:STDOUT:     %C.ref.loc17_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign: <function> = fn_decl @TestAssign [template] {
// CHECK:STDOUT:     %C.ref.loc21_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc21: type = ptr_type C [template = constants.%.8]
// CHECK:STDOUT:     %a.loc21_15.1: C* = param a
// CHECK:STDOUT:     @TestAssign.%a: C* = bind_name a, %a.loc21_15.1
// CHECK:STDOUT:     %C.ref.loc21_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc21_22.1: C = param b
// CHECK:STDOUT:     @TestAssign.%b: C = bind_name b, %b.loc21_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Sub {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in Sub> = import_ref ir1, inst+20, used [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, used [template = imports.%Op.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SubAssign {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in SubAssign> = import_ref ir1, inst+40, used [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+24, unused
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+38, used [template = imports.%Op.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as Sub {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %C.ref.loc9_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: C = param self
// CHECK:STDOUT:     %self.loc9_9.2: C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %C.ref.loc9_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_18.1: C = param other
// CHECK:STDOUT:     %other.loc9_18.2: C = bind_name other, %other.loc9_18.1
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: C as SubAssign {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.3 [template] {
// CHECK:STDOUT:     %C.ref.loc14_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc14_21: type = ptr_type C [template = constants.%.8]
// CHECK:STDOUT:     %self.loc14_14.1: C* = param self
// CHECK:STDOUT:     %self.loc14_14.3: C* = bind_name self, %self.loc14_14.1
// CHECK:STDOUT:     %.loc14_9: C* = addr_pattern %self.loc14_14.3
// CHECK:STDOUT:     %C.ref.loc14_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc14_24.1: C = param other
// CHECK:STDOUT:     %other.loc14_24.2: C = bind_name other, %other.loc14_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.10]
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
// CHECK:STDOUT: fn @Op.1[@impl.1.%self.loc9_9.2: C](@impl.1.%other.loc9_18.2: C) -> @impl.1.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init C = class_init (), @impl.1.%return.var [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_13.3: init C = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%.5]
// CHECK:STDOUT:   return %.loc10_13.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr @impl.2.%self.loc14_14.3: C*](@impl.2.%other.loc14_24.2: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.4[addr %self: Self*](%other: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: C, %b: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.1.%.1, element0 [template = @impl.1.%Op]
// CHECK:STDOUT:   %.loc18_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc17: ref C = splice_block %return {}
// CHECK:STDOUT:   %.loc18_12.2: init C = call %.loc18_12.1(%a.ref, %b.ref) to %.loc17
// CHECK:STDOUT:   return %.loc18_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a: C*, %b: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C* = name_ref a, %a
// CHECK:STDOUT:   %.loc22_3.1: ref C = deref %a.ref
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+22, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+38, unused
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.2.%.1, element0 [template = @impl.2.%Op]
// CHECK:STDOUT:   %.loc22_6.1: <bound method> = bound_method %.loc22_3.1, %.1
// CHECK:STDOUT:   %.loc22_3.2: C* = addr_of %.loc22_3.1
// CHECK:STDOUT:   %.loc22_6.2: init () = call %.loc22_6.1(%.loc22_3.2, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @Sub [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @Sub, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in Sub> = assoc_entity element0, @TestOp.%import_ref.2 [template]
// CHECK:STDOUT:   %.7: type = ptr_type D [template]
// CHECK:STDOUT:   %.8: type = interface_type @SubAssign [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type @SubAssign, <function> [template]
// CHECK:STDOUT:   %.10: <associated <function> in SubAssign> = assoc_entity element0, @TestAssign.%import_ref.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestOp = %TestOp
// CHECK:STDOUT:     .TestAssign = %TestAssign
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestOp: <function> = fn_decl @TestOp [template] {
// CHECK:STDOUT:     %D.ref.loc8_14: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc8_11.1: D = param a
// CHECK:STDOUT:     @TestOp.%a: D = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     %D.ref.loc8_20: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc8_17.1: D = param b
// CHECK:STDOUT:     @TestOp.%b: D = bind_name b, %b.loc8_17.1
// CHECK:STDOUT:     %D.ref.loc8_26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     @TestOp.%return: ref D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAssign: <function> = fn_decl @TestAssign [template] {
// CHECK:STDOUT:     %D.ref.loc15_18: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc15: type = ptr_type D [template = constants.%.7]
// CHECK:STDOUT:     %a.loc15_15.1: D* = param a
// CHECK:STDOUT:     @TestAssign.%a: D* = bind_name a, %a.loc15_15.1
// CHECK:STDOUT:     %D.ref.loc15_25: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc15_22.1: D = param b
// CHECK:STDOUT:     @TestAssign.%b: D = bind_name b, %b.loc15_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Sub {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in Sub> = import_ref ir1, inst+20, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SubAssign {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in SubAssign> = import_ref ir1, inst+40, used [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+24, unused
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+38, unused
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: D, %b: D) -> %return: D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %Sub.decl: invalid = interface_decl @Sub [template = constants.%.4] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a: D*, %b: D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D* = name_ref a, %a
// CHECK:STDOUT:   %.loc19: ref D = deref %a.ref
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+22, used [template = constants.%.8]
// CHECK:STDOUT:   %SubAssign.decl: invalid = interface_decl @SubAssign [template = constants.%.8] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+38, unused
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl_for_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Sub [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: C = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (@impl.%Op) [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @Sub, <function> [template]
// CHECK:STDOUT:   %.8: <associated <function> in Sub> = assoc_entity element0, @TestRhsBad.%import_ref.2 [template]
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
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %import_ref: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:     %Sub.decl: invalid = interface_decl @Sub [template = constants.%.2] {}
// CHECK:STDOUT:     %Sub.ref: type = name_ref Sub, %import_ref [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestRhsBad: <function> = fn_decl @TestRhsBad [template] {
// CHECK:STDOUT:     %C.ref.loc16_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc16_15.1: C = param a
// CHECK:STDOUT:     @TestRhsBad.%a: C = bind_name a, %a.loc16_15.1
// CHECK:STDOUT:     %D.ref.loc16: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %b.loc16_21.1: D = param b
// CHECK:STDOUT:     @TestRhsBad.%b: D = bind_name b, %b.loc16_21.1
// CHECK:STDOUT:     %C.ref.loc16_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestRhsBad.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestLhsBad: <function> = fn_decl @TestLhsBad [template] {
// CHECK:STDOUT:     %D.ref.loc26: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc26_15.1: D = param a
// CHECK:STDOUT:     @TestLhsBad.%a: D = bind_name a, %a.loc26_15.1
// CHECK:STDOUT:     %C.ref.loc26_24: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc26_21.1: C = param b
// CHECK:STDOUT:     @TestLhsBad.%b: C = bind_name b, %b.loc26_21.1
// CHECK:STDOUT:     %C.ref.loc26_30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestLhsBad.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Sub {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in Sub> = import_ref ir1, inst+20, used [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, used [template = imports.%Op]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Sub {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %C.ref.loc9_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: C = param self
// CHECK:STDOUT:     %self.loc9_9.2: C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %C.ref.loc9_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc9_18.1: C = param other
// CHECK:STDOUT:     %other.loc9_18.2: C = bind_name other, %other.loc9_18.1
// CHECK:STDOUT:     %C.ref.loc9_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.6]
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc9_9.2: C](@impl.%other.loc9_18.2: C) -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init C = class_init (), @impl.%return.var [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_13.3: init C = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%.5]
// CHECK:STDOUT:   return %.loc10_13.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRhsBad(%a: C, %b: D) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %b.ref: D = name_ref b, %b
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Op]
// CHECK:STDOUT:   %.loc23_12.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc23_12.2: ref C = temporary_storage
// CHECK:STDOUT:   %.loc23_12.3: init C = call %.loc23_12.1(<invalid>)
// CHECK:STDOUT:   return %.loc23_12.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestLhsBad(%a: D, %b: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
