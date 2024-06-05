// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/mod.carbon

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Mod {
  fn Op[self: C](other: C) -> C {
    return {};
  }
}
impl C as Core.ModAssign {
  fn Op[addr self: C*](other: C) {}
}

fn TestOp(a: C, b: C) -> C {
  return a % b;
}

fn TestAssign(a: C*, b: C) {
  *a %= b;
}

// CHECK:STDOUT: --- mod.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Mod [template]
// CHECK:STDOUT:   %Self.1: Mod = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Op.1: <type of Op> = struct_value () [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: <type of Op> = struct_value () [template]
// CHECK:STDOUT:   %.4: <witness> = interface_witness (%Op.1) [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct: C = struct_value () [template]
// CHECK:STDOUT:   %.6: type = interface_type @ModAssign [template]
// CHECK:STDOUT:   %Self.2: ModAssign = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type C [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.3: <type of Op> = struct_value () [template]
// CHECK:STDOUT:   %.8: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.4: <type of Op> = struct_value () [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Op.3) [template]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [template]
// CHECK:STDOUT:   %TestOp: <type of TestOp> = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type @Mod, <type of Op> [template]
// CHECK:STDOUT:   %.11: <associated <type of Op> in Mod> = assoc_entity element0, file.%import_ref.10 [template]
// CHECK:STDOUT:   %TestAssign.type: type = fn_type @TestAssign [template]
// CHECK:STDOUT:   %TestAssign: <type of TestAssign> = struct_value () [template]
// CHECK:STDOUT:   %.12: type = assoc_entity_type @ModAssign, <type of Op> [template]
// CHECK:STDOUT:   %.13: <associated <type of Op> in ModAssign> = assoc_entity element0, file.%import_ref.12 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:     .TestAssign = %TestAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir4, inst+245, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir4, inst+247, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated <type of Op> in Mod> = import_ref ir4, inst+267, loaded [template = constants.%.11]
// CHECK:STDOUT:   %import_ref.4: <type of Op> = import_ref ir4, inst+263, loaded [template = constants.%Op.2]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc17: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc17: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Mod.ref: type = name_ref Mod, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir4, inst+269, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.6 = import_ref ir4, inst+271, unloaded
// CHECK:STDOUT:   %import_ref.7: <associated <type of Op> in ModAssign> = import_ref ir4, inst+290, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.8: <type of Op> = import_ref ir4, inst+286, loaded [template = constants.%Op.4]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc22: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc22: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %ModAssign.ref: type = name_ref ModAssign, %import_ref.5 [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestOp.decl: <type of TestOp> = fn_decl @TestOp [template = constants.%TestOp] {
// CHECK:STDOUT:     %C.ref.loc26_14: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %a.loc26_11.1: C = param a
// CHECK:STDOUT:     @TestOp.%a: C = bind_name a, %a.loc26_11.1
// CHECK:STDOUT:     %C.ref.loc26_20: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc26_17.1: C = param b
// CHECK:STDOUT:     @TestOp.%b: C = bind_name b, %b.loc26_17.1
// CHECK:STDOUT:     %C.ref.loc26_26: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     @TestOp.%return: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir4, inst+245, loaded [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir4, inst+263, unloaded
// CHECK:STDOUT:   %TestAssign.decl: <type of TestAssign> = fn_decl @TestAssign [template = constants.%TestAssign] {
// CHECK:STDOUT:     %C.ref.loc30_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc30: type = ptr_type C [template = constants.%.7]
// CHECK:STDOUT:     %a.loc30_15.1: C* = param a
// CHECK:STDOUT:     @TestAssign.%a: C* = bind_name a, %a.loc30_15.1
// CHECK:STDOUT:     %C.ref.loc30_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc30_22.1: C = param b
// CHECK:STDOUT:     @TestAssign.%b: C = bind_name b, %b.loc30_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir4, inst+269, loaded [template = constants.%.6]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir4, inst+286, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Mod {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Op = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ModAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.6
// CHECK:STDOUT:   .Op = file.%import_ref.7
// CHECK:STDOUT:   witness = (file.%import_ref.8)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as Mod {
// CHECK:STDOUT:   %Op.decl: <type of Op> = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %C.ref.loc18_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.loc18_9.1: C = param self
// CHECK:STDOUT:     %self.loc18_9.2: C = bind_name self, %self.loc18_9.1
// CHECK:STDOUT:     %C.ref.loc18_25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc18_18.1: C = param other
// CHECK:STDOUT:     %other.loc18_18.2: C = bind_name other, %other.loc18_18.1
// CHECK:STDOUT:     %C.ref.loc18_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.var: ref C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: C as ModAssign {
// CHECK:STDOUT:   %Op.decl: <type of Op> = fn_decl @Op.3 [template = constants.%Op.3] {
// CHECK:STDOUT:     %C.ref.loc23_20: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc23_21: type = ptr_type C [template = constants.%.7]
// CHECK:STDOUT:     %self.loc23_14.1: C* = param self
// CHECK:STDOUT:     %self.loc23_14.3: C* = bind_name self, %self.loc23_14.1
// CHECK:STDOUT:     %.loc23_9: C* = addr_pattern %self.loc23_14.3
// CHECK:STDOUT:     %C.ref.loc23_31: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %other.loc23_24.1: C = param other
// CHECK:STDOUT:     %other.loc23_24.2: C = bind_name other, %other.loc23_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (%Op.decl) [template = constants.%.9]
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
// CHECK:STDOUT: fn @Op.1[@impl.1.%self.loc18_9.2: C](@impl.1.%other.loc18_18.2: C) -> @impl.1.%return.var: C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc19_13.2: init C = class_init (), @impl.1.%return.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_14: init C = converted %.loc19_13.1, %.loc19_13.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc19_14 to @impl.1.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self: Self](%other: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.3[addr @impl.2.%self.loc23_14.3: C*](@impl.2.%other.loc23_24.2: C) {
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
// CHECK:STDOUT:   %.1: <type of Op> = interface_witness_access @impl.1.%.1, element0 [template = constants.%Op.1]
// CHECK:STDOUT:   %.loc27: <bound method> = bound_method %a.ref, %.1
// CHECK:STDOUT:   %.loc26: ref C = splice_block %return {}
// CHECK:STDOUT:   %Op.call: init C = call %.loc27(%a.ref, %b.ref) to %.loc26
// CHECK:STDOUT:   return %Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAssign(%a: C*, %b: C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: C* = name_ref a, %a
// CHECK:STDOUT:   %.loc31_3.1: ref C = deref %a.ref
// CHECK:STDOUT:   %b.ref: C = name_ref b, %b
// CHECK:STDOUT:   %.1: <type of Op> = interface_witness_access @impl.2.%.1, element0 [template = constants.%Op.3]
// CHECK:STDOUT:   %.loc31_6: <bound method> = bound_method %.loc31_3.1, %.1
// CHECK:STDOUT:   %.loc31_3.2: C* = addr_of %.loc31_3.1
// CHECK:STDOUT:   %Op.call: init () = call %.loc31_6(%.loc31_3.2, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
