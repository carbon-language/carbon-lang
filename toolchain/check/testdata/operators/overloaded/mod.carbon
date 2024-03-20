// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This file was generated from binary_op.carbon.tmpl. Run make_tests.sh to regenerate.

// --- prelude.carbon

package Core api;

interface Mod {
  fn Op[self: Self](other: Self) -> Self;
}
interface ModAssign {
  fn Op[addr self: Self*](other: Self);
}

// --- user.carbon

package User api;

import Core;

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

// CHECK:STDOUT: --- prelude.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Mod [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Mod, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Mod> = assoc_entity element0, @Mod.%Op [template]
// CHECK:STDOUT:   %.4: type = interface_type @ModAssign [template]
// CHECK:STDOUT:   %.5: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @ModAssign, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in ModAssign> = assoc_entity element0, @ModAssign.%Op [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .ModAssign = %ModAssign.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod.decl: type = interface_decl @Mod [template = constants.%.1] {}
// CHECK:STDOUT:   %ModAssign.decl: type = interface_decl @ModAssign [template = constants.%.4] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Mod {
// CHECK:STDOUT:   %Self: Mod = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc5_15: Mod = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.1: type = facet_type_access %Self.ref.loc5_15 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_15.2: type = converted %Self.ref.loc5_15, %.loc5_15.1 [symbolic = %Self]
// CHECK:STDOUT:     %self.loc5_9.1: Self = param self
// CHECK:STDOUT:     %self.loc5_9.2: Self = bind_name self, %self.loc5_9.1
// CHECK:STDOUT:     %Self.ref.loc5_28: Mod = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_28.1: type = facet_type_access %Self.ref.loc5_28 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_28.2: type = converted %Self.ref.loc5_28, %.loc5_28.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc5_21.1: Self = param other
// CHECK:STDOUT:     %other.loc5_21.2: Self = bind_name other, %other.loc5_21.1
// CHECK:STDOUT:     %Self.ref.loc5_37: Mod = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_37.1: type = facet_type_access %Self.ref.loc5_37 [symbolic = %Self]
// CHECK:STDOUT:     %.loc5_37.2: type = converted %Self.ref.loc5_37, %.loc5_37.1 [symbolic = %Self]
// CHECK:STDOUT:     %return.var: ref Self = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_41: <associated <function> in Mod> = assoc_entity element0, %Op [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc5_41
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ModAssign {
// CHECK:STDOUT:   %Self: ModAssign = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Op: <function> = fn_decl @Op.2 [template] {
// CHECK:STDOUT:     %Self.ref.loc8_20: ModAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.1: type = facet_type_access %Self.ref.loc8_20 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_20: type = converted %Self.ref.loc8_20, %.loc8_24.1 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_24.2: type = ptr_type Self [symbolic = constants.%.5]
// CHECK:STDOUT:     %self.loc8_14.1: Self* = param self
// CHECK:STDOUT:     %self.loc8_14.3: Self* = bind_name self, %self.loc8_14.1
// CHECK:STDOUT:     %.loc8_9: Self* = addr_pattern %self.loc8_14.3
// CHECK:STDOUT:     %Self.ref.loc8_34: ModAssign = name_ref Self, %Self [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.1: type = facet_type_access %Self.ref.loc8_34 [symbolic = %Self]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %Self.ref.loc8_34, %.loc8_34.1 [symbolic = %Self]
// CHECK:STDOUT:     %other.loc8_27.1: Self = param other
// CHECK:STDOUT:     %other.loc8_27.2: Self = bind_name other, %other.loc8_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_39: <associated <function> in ModAssign> = assoc_entity element0, %Op [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8_39
// CHECK:STDOUT:   witness = (%Op)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.1[@Mod.%self.loc5_9.2: Self](@Mod.%other.loc5_21.2: Self) -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr @ModAssign.%self.loc8_14.3: Self*](@ModAssign.%other.loc8_27.2: Self);
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = interface_type @Mod [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: C = struct_value () [template]
// CHECK:STDOUT:   %.6: <witness> = interface_witness (@impl.1.%Op) [template]
// CHECK:STDOUT:   %.7: type = interface_type @ModAssign [template]
// CHECK:STDOUT:   %.8: type = ptr_type C [template]
// CHECK:STDOUT:   %.9: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (@impl.2.%Op) [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type @Mod, <function> [template]
// CHECK:STDOUT:   %.12: <associated <function> in Mod> = assoc_entity element0, file.%import_ref.10 [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type @ModAssign, <function> [template]
// CHECK:STDOUT:   %.14: <associated <function> in ModAssign> = assoc_entity element0, file.%import_ref.12 [template]
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
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.2: <associated <function> in Mod> = import_ref ir1, inst+20, used [template = constants.%.12]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+3, unused
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%Op.1]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc8: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %Mod.decl: invalid = interface_decl @Mod [template = constants.%.2] {}
// CHECK:STDOUT:     %Mod.ref: type = name_ref Mod, %import_ref.1 [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: type = import_ref ir1, inst+22, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.6: <associated <function> in ModAssign> = import_ref ir1, inst+40, used [template = constants.%.14]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir1, inst+24, unused
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir1, inst+38, used [template = imports.%Op.2]
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %Core.ref.loc13: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:     %ModAssign.decl: invalid = interface_decl @ModAssign [template = constants.%.7] {}
// CHECK:STDOUT:     %ModAssign.ref: type = name_ref ModAssign, %import_ref.5 [template = constants.%.7]
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
// CHECK:STDOUT:   %import_ref.9: type = import_ref ir1, inst+1, used [template = constants.%.2]
// CHECK:STDOUT:   %import_ref.10 = import_ref ir1, inst+18, unused
// CHECK:STDOUT:   %TestAssign: <function> = fn_decl @TestAssign [template] {
// CHECK:STDOUT:     %C.ref.loc21_18: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc21: type = ptr_type C [template = constants.%.8]
// CHECK:STDOUT:     %a.loc21_15.1: C* = param a
// CHECK:STDOUT:     @TestAssign.%a: C* = bind_name a, %a.loc21_15.1
// CHECK:STDOUT:     %C.ref.loc21_25: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %b.loc21_22.1: C = param b
// CHECK:STDOUT:     @TestAssign.%b: C = bind_name b, %b.loc21_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.11: type = import_ref ir1, inst+22, used [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.12 = import_ref ir1, inst+38, unused
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Mod {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.2
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ModAssign {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = file.%import_ref.6
// CHECK:STDOUT:   .Self = file.%import_ref.7
// CHECK:STDOUT:   witness = (file.%import_ref.8)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: C as Mod {
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
// CHECK:STDOUT: impl @impl.2: C as ModAssign {
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
// CHECK:STDOUT:   %.1: <function> = interface_witness_access @impl.2.%.1, element0 [template = @impl.2.%Op]
// CHECK:STDOUT:   %.loc22_6.1: <bound method> = bound_method %.loc22_3.1, %.1
// CHECK:STDOUT:   %.loc22_3.2: C* = addr_of %.loc22_3.1
// CHECK:STDOUT:   %.loc22_6.2: init () = call %.loc22_6.1(%.loc22_3.2, %b.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
