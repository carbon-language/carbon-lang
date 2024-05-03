// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

// --- prelude.carbon

package Core api;

interface BitComplement {
  fn Op[self: Self]() -> Self;
}

// --- user.carbon

package User api;

import Core;

class C {};

impl C as Core.BitComplement {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  return ^a;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @BitComplement [template]
// CHECK:STDOUT:   %Self: BitComplement = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op: type = fn_type @Op [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Op = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @BitComplement, Op [template]
// CHECK:STDOUT:   %.4: <associated Op in BitComplement> = assoc_entity element0, @BitComplement.%Op.decl [template]
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
// CHECK:STDOUT:   %Self: BitComplement = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op [template = constants.%struct] {
// CHECK:STDOUT:     %Self.ref.loc5_15: BitComplement = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_26: BitComplement = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_26.1: type = facet_type_access %Self.ref.loc5_26 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %Self.ref.loc5_26, %.loc5_26.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_30: <associated Op in BitComplement> = assoc_entity element0, %Op.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_30
// CHECK:STDOUT:   witness = (%Op.decl)
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
// CHECK:STDOUT:   %Self: BitComplement = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Op = struct_value () [template]
// CHECK:STDOUT:   %Op.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %struct.2: Op = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%struct.1) [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %TestOp: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %struct.4: TestOp = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @BitComplement, Op [template]
// CHECK:STDOUT:   %.7: <associated Op in BitComplement> = assoc_entity element0, file.%import_ref.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated Op in BitComplement> = import_ref ir2, inst+19, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: Op = import_ref ir2, inst+14, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %BitComplement.ref: type = name_ref BitComplement, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: TestOp = fn_decl @TestOp [template = constants.%struct.4] {
// CHECK:STDOUT:     %C.ref.loc14_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc14_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc14_11.1
// CHECK:STDOUT:     %C.ref.loc14_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir2, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir2, inst+14, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitComplement {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as BitComplement {
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc9_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: C = param self
// CHECK:STDOUT:     %self.loc9_9.2: C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %C.ref.loc9_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc9_9.2: C]() -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init C = class_init (), @impl.%return.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc10_14: init C = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%struct.3]
// CHECK:STDOUT:   return %.loc10_14 to @impl.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %.1: Op = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc15: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc14: ref C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init C = call %.loc15(%a.ref) to %.loc14
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
