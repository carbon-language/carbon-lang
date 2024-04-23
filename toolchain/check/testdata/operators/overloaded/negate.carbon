// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from unary_op.carbon.tmpl. Run make_tests.sh to regenerate.

// --- prelude.carbon

package Core api;

interface Negate {
  fn Op[self: Self]() -> Self;
}

// --- user.carbon

package User api;

import Core;

class C {};

impl C as Core.Negate {
  fn Op[self: C]() -> C {
    return {};
  }
}

fn TestOp(a: C) -> C {
  return -a;
}

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: Negate = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Negate, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Negate> = assoc_entity element0, @Negate.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: type = interface_decl @Negate [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT:   %Self: Negate = bind_symbolic_name Self [symbolic = constants.%Self]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op [template] {
// CHECK:STDOUT:     %Self.ref.loc5_15: Negate = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_26: Negate = name_ref Self, %Self [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_26.1: type = facet_type_access %Self.ref.loc5_26 [symbolic = constants.%Self]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %Self.ref.loc5_26, %.loc5_26.1 [symbolic = constants.%Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_30: <associated <function> in Negate> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_30
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op[@Negate.%self.loc5_9.2: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: Negate = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %.3: <witness> = interface_witness (@impl.%Op) [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: C = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @Negate, <function> [template]
// CHECK:STDOUT:   %.8: <associated <function> in Negate> = assoc_entity element0, file.%import_ref.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir3, inst+1, loc_18 [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Negate> = import_ref ir3, inst+16, loc_50 [template = constants.%.8]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir3, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir3, inst+14, loc_19 [template = imports.%Op]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Negate.decl: type = interface_decl @Negate [template = constants.%.2] {}
// CHECK:STDOUT:     %Negate.ref: type = name_ref Negate, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp: <function> = fn_decl @TestOp [template] {
// CHECK:STDOUT:     %C.ref.loc14_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc14_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc14_11.1
// CHECK:STDOUT:     %C.ref.loc14_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir3, inst+1, loc_50 [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir3, inst+14, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: C as Negate {
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %C.ref.loc9_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc9_9.1: C = param self
// CHECK:STDOUT:     %self.loc9_9.2: C = bind_name self, %self.loc9_9.1
// CHECK:STDOUT:     %C.ref.loc9_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op) [template = constants.%.3]
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
// CHECK:STDOUT: fn @Op.1[@impl.%self.loc9_9.2: C]() -> @impl.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc10_13.2: init C = class_init (), @impl.%return.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_14: init C = converted %.loc10_13.1, %.loc10_13.2 [template = constants.%.6]
// CHECK:STDOUT:   return %.loc10_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp(%a: C) -> %return: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C = name_ref a, %a
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.%.1, element0 [template = @impl.%Op]
// CHECK:STDOUT:   %.loc15_10.1: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc14: ref C = splice_block %return {}
// CHECK:STDOUT:   %.loc15_10.2: init C = call %.loc15_10.1(%a.ref) to %.loc14
// CHECK:STDOUT:   return %.loc15_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
