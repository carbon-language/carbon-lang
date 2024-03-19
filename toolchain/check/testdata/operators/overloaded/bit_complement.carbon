// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from unary_op.carbon.in. Run make_tests.sh to regenerate.

// --- prelude.carbon

package Carbon api;

interface BitComplement {
  fn Op[self: Self]() -> Self;
}

// --- user.carbon

package User api;

import Carbon;

class C {};

impl C as Carbon.BitComplement {
  // CHECK:STDERR: user.carbon:[[@LINE+4]]:3: ERROR: Function redeclaration differs because of parameter count of 1.
  // CHECK:STDERR:   fn Op[self: C](other: C) -> C {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: user.carbon: Previously declared with parameter count of 0.
  fn Op[self: C](other: C) -> C {
    return {};
  }
}

fn TestOp(a: C, b: C) -> C {
  return ^a;
}

// --- fail_no_impl.carbon

package FailNoImpl api;

import Carbon;

class D {};

fn TestOp(a: D) {
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+9]]:3: ERROR: No return expression should be provided in this context.
  // CHECK:STDERR:   return ^a;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE-4]]:1: There was no return type provided.
  // CHECK:STDERR: fn TestOp(a: D) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_no_impl.carbon:[[@LINE+3]]:10: ERROR: Cannot access member of interface BitComplement in type D that does not implement that interface.
  // CHECK:STDERR:   return ^a;
  // CHECK:STDERR:          ^~
  return ^a;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @BitComplement [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @BitComplement, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in BitComplement> = assoc_entity element0, @BitComplement.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .BitComplement = %BitComplement.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BitComplement.decl: type = interface_decl @BitComplement [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitComplement {
// CHECK:STDOUT:   %Self: BitComplement = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op [template] {
// CHECK:STDOUT:     %Self.ref.loc5_15: BitComplement = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_26: BitComplement = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_26.1: type = facet_type_access %Self.ref.loc5_26 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %Self.ref.loc5_26, %.loc5_26.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_30: <associated <function> in BitComplement> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_30
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op[@BitComplement.%self.loc5_9.2: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @BitComplement [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: C = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @BitComplement, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in BitComplement> = assoc_entity element0, @TestOp.%import_ref.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Carbon = %Carbon
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Carbon: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Carbon.ref: <namespace> = name_ref Carbon, %Carbon [template = %Carbon]
// CHECK:STDOUT:     %import_ref: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:     %BitComplement.decl: invalid = interface_decl @BitComplement [template = constants.%.2] {}
// CHECK:STDOUT:     %BitComplement.ref: type = name_ref BitComplement, %import_ref [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp: <function> = fn_decl @TestOp [template] {
// CHECK:STDOUT:     %C.ref.loc18_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc18_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc18_11.1
// CHECK:STDOUT:     %C.ref.loc18_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc18_17.1: C = param b
// CHECK:STDOUT:     @TestOp.%b: C = bind_name b, %b.loc18_17.1
// CHECK:STDOUT:     %C.ref.loc18_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitComplement {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in BitComplement> = import_ref ir1, inst+15, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+13, used [template = imports.%Op]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %import_ref.1
// CHECK:STDOUT:   .Self = %import_ref.2
// CHECK:STDOUT:   witness = (%import_ref.3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as BitComplement {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %C.ref.loc13_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc13_9.1: C = param self
// CHECK:STDOUT:     %self.loc13_9.2: C = bind_name self, %self.loc13_9.1
// CHECK:STDOUT:     %C.ref.loc13_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc13_18.1: C = param other
// CHECK:STDOUT:     %other.loc13_18.2: C = bind_name other, %other.loc13_18.1
// CHECK:STDOUT:     %C.ref.loc13_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
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
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc13_9.2: C](@impl.%other.loc13_18.2: C) -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc14_13.2: init C = class_init (), @impl.%return.var [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_13.3: init C = converted %.loc14_13.1, %.loc14_13.2 [template = constants.%.5]
// CHECK:STDOUT:   return %.loc14_13.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: C, %b: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+13, unused
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: type = interface_type @BitComplement [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @BitComplement, <function> [template]
// CHECK:STDOUT:   %.6: <associated <function> in BitComplement> = assoc_entity element0, @TestOp.%import_ref.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Carbon = %Carbon
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .TestOp = %TestOp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Carbon: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %TestOp: <function> = fn_decl @TestOp [template] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %a.loc8_11.1: D = param a
// CHECK:STDOUT:     @TestOp.%a: D = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitComplement {
// CHECK:STDOUT:   %import_ref.1: <associated <function> in BitComplement> = import_ref ir1, inst+15, used [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+13, unused
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
// CHECK:STDOUT: fn @TestOp(%a: D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: D = name_ref a, %a
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.4]
// CHECK:STDOUT:   %BitComplement.decl: invalid = interface_decl @BitComplement [template = constants.%.4] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+13, unused
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
