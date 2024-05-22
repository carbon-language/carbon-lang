// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.BitComplement {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  return ^a;
}

// CHECK:STDOUT: --- bit_complement.carbon
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir6, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Op in BitComplement> = import_ref ir6, inst+19, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.4: Op = import_ref ir6, inst+14, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %BitComplement.ref: type = name_ref BitComplement, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: TestOp = fn_decl @TestOp [template = constants.%struct.4] {
// CHECK:STDOUT:     %C.ref.loc19_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc19_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc19_11.1
// CHECK:STDOUT:     %C.ref.loc19_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir6, inst+1, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir6, inst+14, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BitComplement {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Op = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as BitComplement {
// CHECK:STDOUT:   %Op.decl: Op = fn_decl @Op.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %C.ref.loc14_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc14_9.1: C = param self
// CHECK:STDOUT:     %self.loc14_9.2: C = bind_name self, %self.loc14_9.1
// CHECK:STDOUT:     %C.ref.loc14_23: type = name_ref C, file.%C.decl [template = constants.%C]
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
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc14_9.2: C]() -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: init C = class_init (), @impl.%return.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc15_14: init C = converted %.loc15_13.1, %.loc15_13.2 [template = constants.%struct.3]
// CHECK:STDOUT:   return %.loc15_14 to @impl.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %.1: Op = interface_witness_access @impl.%.1, element0 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc19: ref C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init C = call %.loc20(%a.ref) to %.loc19
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
