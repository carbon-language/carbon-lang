// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/no_prelude/impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/no_prelude/impl_forall.carbon

// --- impl_forall.carbon
library "[[@TEST_NAME]]";

class A(T:! type) {
  var n: T;
}

interface I(U:! type) {
  fn F[self: Self]() -> U;
}

impl forall [V:! type] A(V) as I(V) {
  fn F[self: Self]() -> V {
    return self.n;
  }
}

fn TestGeneric[W:! type](a: A(W)) -> W {
  return a.(I(W).F)();
}

fn TestSpecific(a: A({})) -> {} {
  return a.(I({}).F)();
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.13025a.1: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.1: type = unbound_element_type %A.13025a.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.1: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.1: <witness> = complete_type_witness %struct_type.n.848971.1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325e65.1: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.1: %F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955255.1: type = assoc_entity_type %I.type.325e65.1 [symbolic]
// CHECK:STDOUT:   %assoc0.fef501.1: %I.assoc_type.955255.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %A.13025a.2: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.325e65.2: type = facet_type <@I, @I(%V)> [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.2: type = fn_type @F.1, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.2: %F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955255.2: type = assoc_entity_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %assoc0.fef501.2: %I.assoc_type.955255.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfebb2.1: <witness> = require_complete_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %impl_witness.ab3b51.1: <witness> = impl_witness (@impl.%F.decl), @impl(%V) [symbolic]
// CHECK:STDOUT:   %F.type.0fea45.1: type = fn_type @F.2, @impl(%V) [symbolic]
// CHECK:STDOUT:   %F.d6ae34.1: %F.type.0fea45.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.5b0ece.1: %I.type.325e65.2 = facet_value %A.13025a.2, %impl_witness.ab3b51.1 [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.2: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.2: type = unbound_element_type %A.13025a.2, %V [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.2: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.2: <witness> = complete_type_witness %struct_type.n.848971.2 [symbolic]
// CHECK:STDOUT:   %require_complete.5b8aee.1: <witness> = require_complete_type %A.13025a.2 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %A.13025a.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [concrete]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4aeca8.3: <witness> = require_complete_type %W [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.3: type = unbound_element_type %A.13025a.3, %W [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.3: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.3: <witness> = complete_type_witness %struct_type.n.848971.3 [symbolic]
// CHECK:STDOUT:   %require_complete.5b8aee.2: <witness> = require_complete_type %A.13025a.3 [symbolic]
// CHECK:STDOUT:   %I.type.325e65.3: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.3: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.3: %F.type.2aef59.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.955255.3: type = assoc_entity_type %I.type.325e65.3 [symbolic]
// CHECK:STDOUT:   %assoc0.fef501.3: %I.assoc_type.955255.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfebb2.2: <witness> = require_complete_type %I.type.325e65.3 [symbolic]
// CHECK:STDOUT:   %impl_witness.ab3b51.2: <witness> = impl_witness (@impl.%F.decl), @impl(%W) [symbolic]
// CHECK:STDOUT:   %F.type.0fea45.2: type = fn_type @F.2, @impl(%W) [symbolic]
// CHECK:STDOUT:   %F.d6ae34.2: %F.type.0fea45.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.5b0ece.2: %I.type.325e65.3 = facet_value %A.13025a.3, %impl_witness.ab3b51.2 [symbolic]
// CHECK:STDOUT:   %.051: type = fn_type_with_self_type %F.type.2aef59.3, %I.facet.5b0ece.2 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.e98: <specific function> = specific_function %F.d6ae34.2, @F.2(%W) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A.235: type = class_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [concrete]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.elem.2af: type = unbound_element_type %A.235, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.n.91c: type = struct_type {.n: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.n.91c [concrete]
// CHECK:STDOUT:   %I.type.885: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %F.type.684: type = fn_type @F.1, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.a8d: %F.type.684 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.67f: type = assoc_entity_type %I.type.885 [concrete]
// CHECK:STDOUT:   %assoc0.639: %I.assoc_type.67f = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %complete_type.788: <witness> = complete_type_witness %I.type.885 [concrete]
// CHECK:STDOUT:   %impl_witness.f35: <witness> = impl_witness (@impl.%F.decl), @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.type.875: type = fn_type @F.2, @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.158: %F.type.875 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.f67: %I.type.885 = facet_value %A.235, %impl_witness.f35 [concrete]
// CHECK:STDOUT:   %.f01: type = fn_type_with_self_type %F.type.684, %I.facet.f67 [concrete]
// CHECK:STDOUT:   %F.specific_fn.72f: <specific function> = specific_function %F.158, @F.2(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .TestGeneric = %TestGeneric.decl
// CHECK:STDOUT:     .TestSpecific = %TestSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc3_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %U.patt.loc7_13.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc7_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {
// CHECK:STDOUT:     %V.patt.loc11_14.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc11_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc11_26: type = name_ref V, %V.loc11_14.1 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:     %A.loc11_27.1: type = class_type @A, @A(constants.%V) [symbolic = %A.loc11_27.2 (constants.%A.13025a.2)]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc11_34: type = name_ref V, %V.loc11_14.1 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc11_35.1: type = facet_type <@I, @I(constants.%V)> [symbolic = %I.type.loc11_35.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     %V.loc11_14.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%V) [symbolic = @impl.%impl_witness (constants.%impl_witness.ab3b51.1)]
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [concrete = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt.loc17_16.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc17_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %a.patt: @TestGeneric.%A.loc17_32.2 (%A.13025a.3) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @TestGeneric.%A.loc17_32.2 (%A.13025a.3) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: @TestGeneric.%W.loc17_16.2 (%W) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @TestGeneric.%W.loc17_16.2 (%W) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.ref.loc17_38: type = name_ref W, %W.loc17_16.1 [symbolic = %W.loc17_16.2 (constants.%W)]
// CHECK:STDOUT:     %W.loc17_16.1: type = bind_symbolic_name W, 0 [symbolic = %W.loc17_16.2 (constants.%W)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.loc17_32.2 (%A.13025a.3) = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %A.loc17_32.1 [symbolic = %A.loc17_32.2 (constants.%A.13025a.3)] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %W.ref.loc17_31: type = name_ref W, %W.loc17_16.1 [symbolic = %W.loc17_16.2 (constants.%W)]
// CHECK:STDOUT:       %A.loc17_32.1: type = class_type @A, @A(constants.%W) [symbolic = %A.loc17_32.2 (constants.%A.13025a.3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @TestGeneric.%A.loc17_32.2 (%A.13025a.3) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @TestGeneric.%W.loc17_16.2 (%W) = out_param call_param1
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.loc17_16.2 (%W) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [concrete = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %A.235 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A.235 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc21_31.2: type = converted %.loc21_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %a.param: %A.235 = value_param call_param0
// CHECK:STDOUT:     %.loc21_24.1: type = splice_block %A [concrete = constants.%A.235] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %.loc21_23: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc21_24.2: type = converted %.loc21_23, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%empty_struct_type) [concrete = constants.%A.235]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %A.235 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc7_13.1: type) {
// CHECK:STDOUT:   %U.loc7_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc7_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc7_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc7_13.2)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self.2: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%U.loc7_13.2) [symbolic = %F.type (constants.%F.type.2aef59.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2aef59.1) = struct_value () [symbolic = %F (constants.%F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.325e65.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.955255.1)]
// CHECK:STDOUT:   %assoc0.loc8_26.2: @I.%I.assoc_type (%I.assoc_type.955255.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc8_26.2 (constants.%assoc0.fef501.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.2aef59.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.1.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%U (%U) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc7_13.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc8_14.1: type = splice_block %.loc8_14.3 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc8_14.2: @F.1.%I.type (%I.type.325e65.1) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.ref: @F.1.%I.type (%I.type.325e65.1) = name_ref Self, %.loc8_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:         %Self.as_type.loc8_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc8_14.3: type = converted %Self.ref, %Self.as_type.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.1.%U (%U) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.1.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_26.1: @I.%I.assoc_type (%I.assoc_type.955255.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc8_26.2 (constants.%assoc0.fef501.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc8_26.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V.loc11_14.1: type) {
// CHECK:STDOUT:   %V.loc11_14.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc11_14.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc11_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %A.loc11_27.2: type = class_type @A, @A(%V.loc11_14.2) [symbolic = %A.loc11_27.2 (constants.%A.13025a.2)]
// CHECK:STDOUT:   %I.type.loc11_35.2: type = facet_type <@I, @I(%V.loc11_14.2)> [symbolic = %I.type.loc11_35.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc11_35.2 (%I.type.325e65.2) [symbolic = %require_complete (constants.%require_complete.cfebb2.1)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%V.loc11_14.2) [symbolic = %impl_witness (constants.%impl_witness.ab3b51.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V.loc11_14.2) [symbolic = %F.type (constants.%F.type.0fea45.1)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.0fea45.1) = struct_value () [symbolic = %F (constants.%F.d6ae34.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc11_27.1 as %I.type.loc11_35.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.0fea45.1) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.d6ae34.1)] {
// CHECK:STDOUT:       %self.patt: @F.2.%A (%A.13025a.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%A (%A.13025a.2) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.2.%V (%V) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%V (%V) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %V.ref: type = name_ref V, @impl.%V.loc11_14.1 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %self.param: @F.2.%A (%A.13025a.2) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%A.loc11_27.1 [symbolic = %A (constants.%A.13025a.2)]
// CHECK:STDOUT:       %self: @F.2.%A (%A.13025a.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%V (%V) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.2.%V (%V) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = file.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc3_9.1: type) {
// CHECK:STDOUT:   %T.loc3_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%T.loc3_9.2 (%T) [symbolic = %require_complete (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc3_9.2) [symbolic = %A (constants.%A.13025a.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type @A.%A (%A.13025a.1), @A.%T.loc3_9.2 (%T) [symbolic = %A.elem (constants.%A.elem.1ceb36.1)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @A.%T.loc3_9.2 (%T)} [symbolic = %struct_type.n (constants.%struct_type.n.848971.1)]
// CHECK:STDOUT:   %complete_type.loc5_1.2: <witness> = complete_type_witness @A.%struct_type.n (%struct_type.n.848971.1) [symbolic = %complete_type.loc5_1.2 (constants.%complete_type.84bb3d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_8: @A.%A.elem (%A.elem.1ceb36.1) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc4_3: @A.%A.elem (%A.elem.1ceb36.1) = var_pattern %.loc4_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @A.%A.elem (%A.elem.1ceb36.1) = var <none>
// CHECK:STDOUT:     %complete_type.loc5_1.1: <witness> = complete_type_witness %struct_type.n.848971.1 [symbolic = %complete_type.loc5_1.2 (constants.%complete_type.84bb3d.1)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc5_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.13025a.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc4_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%U.loc7_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325e65.1)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V.loc11_14.1: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.13025a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_22: <witness> = require_complete_type @F.2.%V (%V) [symbolic = %require_complete.loc12_22 (constants.%require_complete.4aeca8.2)]
// CHECK:STDOUT:   %require_complete.loc12_12: <witness> = require_complete_type @F.2.%A (%A.13025a.2) [symbolic = %require_complete.loc12_12 (constants.%require_complete.5b8aee.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type @F.2.%A (%A.13025a.2), @F.2.%V (%V) [symbolic = %A.elem (constants.%A.elem.1ceb36.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%A (%A.13025a.2)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.13025a.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @F.2.%A.elem (%A.elem.1ceb36.2) = name_ref n, @A.%.loc4_8 [concrete = @A.%.loc4_8]
// CHECK:STDOUT:     %.loc13_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc13_16.2: @F.2.%V (%V) = bind_value %.loc13_16.1
// CHECK:STDOUT:     return %.loc13_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc17_16.1: type) {
// CHECK:STDOUT:   %W.loc17_16.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc17_16.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc17_16.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc17_16.2 (constants.%W.patt)]
// CHECK:STDOUT:   %A.loc17_32.2: type = class_type @A, @A(%W.loc17_16.2) [symbolic = %A.loc17_32.2 (constants.%A.13025a.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_35: <witness> = require_complete_type @TestGeneric.%W.loc17_16.2 (%W) [symbolic = %require_complete.loc17_35 (constants.%require_complete.4aeca8.3)]
// CHECK:STDOUT:   %require_complete.loc17_27: <witness> = require_complete_type @TestGeneric.%A.loc17_32.2 (%A.13025a.3) [symbolic = %require_complete.loc17_27 (constants.%require_complete.5b8aee.2)]
// CHECK:STDOUT:   %I.type.loc18_16.2: type = facet_type <@I, @I(%W.loc17_16.2)> [symbolic = %I.type.loc18_16.2 (constants.%I.type.325e65.3)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type @TestGeneric.%I.type.loc18_16.2 (%I.type.325e65.3) [symbolic = %require_complete.loc18 (constants.%require_complete.cfebb2.2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @TestGeneric.%I.type.loc18_16.2 (%I.type.325e65.3) [symbolic = %I.assoc_type (constants.%I.assoc_type.955255.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%I.assoc_type (%I.assoc_type.955255.3) = assoc_entity element0, @I.%F.decl [symbolic = %assoc0 (constants.%assoc0.fef501.3)]
// CHECK:STDOUT:   %F.type.loc18_11.1: type = fn_type @F.1, @I(%W.loc17_16.2) [symbolic = %F.type.loc18_11.1 (constants.%F.type.2aef59.3)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%W.loc17_16.2) [symbolic = %impl_witness (constants.%impl_witness.ab3b51.2)]
// CHECK:STDOUT:   %I.facet: @TestGeneric.%I.type.loc18_16.2 (%I.type.325e65.3) = facet_value %A.loc17_32.2, %impl_witness [symbolic = %I.facet (constants.%I.facet.5b0ece.2)]
// CHECK:STDOUT:   %.loc18_11: type = fn_type_with_self_type %F.type.loc18_11.1, %I.facet [symbolic = %.loc18_11 (constants.%.051)]
// CHECK:STDOUT:   %F.type.loc18_11.2: type = fn_type @F.2, @impl(%W.loc17_16.2) [symbolic = %F.type.loc18_11.2 (constants.%F.type.0fea45.2)]
// CHECK:STDOUT:   %F: @TestGeneric.%F.type.loc18_11.2 (%F.type.0fea45.2) = struct_value () [symbolic = %F (constants.%F.d6ae34.2)]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F.2(%W.loc17_16.2) [symbolic = %F.specific_fn (constants.%F.specific_fn.e98)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.patt.loc17_16.1: type](%a.param_patt: @TestGeneric.%A.loc17_32.2 (%A.13025a.3)) -> @TestGeneric.%W.loc17_16.2 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.loc17_32.2 (%A.13025a.3) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc18: type = name_ref W, %W.loc17_16.1 [symbolic = %W.loc17_16.2 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc18_16.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc18_16.2 (constants.%I.type.325e65.3)]
// CHECK:STDOUT:     %.loc18_17: @TestGeneric.%I.assoc_type (%I.assoc_type.955255.3) = specific_constant @I.%assoc0.loc8_26.1, @I(constants.%W) [symbolic = %assoc0 (constants.%assoc0.fef501.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%I.assoc_type (%I.assoc_type.955255.3) = name_ref F, %.loc18_17 [symbolic = %assoc0 (constants.%assoc0.fef501.3)]
// CHECK:STDOUT:     %impl.elem0: @TestGeneric.%.loc18_11 (%.051) = impl_witness_access constants.%impl_witness.ab3b51.2, element0 [symbolic = %F (constants.%F.d6ae34.2)]
// CHECK:STDOUT:     %bound_method.loc18_11: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%W) [symbolic = %F.specific_fn (constants.%F.specific_fn.e98)]
// CHECK:STDOUT:     %bound_method.loc18_21: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:     %F.call: init @TestGeneric.%W.loc17_16.2 (%W) = call %bound_method.loc18_21(%a.ref)
// CHECK:STDOUT:     %.loc18_22.1: @TestGeneric.%W.loc17_16.2 (%W) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc18_22.2: @TestGeneric.%W.loc17_16.2 (%W) = converted %F.call, %.loc18_22.1
// CHECK:STDOUT:     return %.loc18_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param_patt: %A.235) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.235 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:   %.loc22_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_17: type = converted %.loc22_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.885]
// CHECK:STDOUT:   %.loc22_18: %I.assoc_type.67f = specific_constant @I.%assoc0.loc8_26.1, @I(constants.%empty_struct_type) [concrete = constants.%assoc0.639]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.67f = name_ref F, %.loc22_18 [concrete = constants.%assoc0.639]
// CHECK:STDOUT:   %impl.elem0: %.f01 = impl_witness_access constants.%impl_witness.f35, element0 [concrete = constants.%F.158]
// CHECK:STDOUT:   %bound_method.loc22_11: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%empty_struct_type) [concrete = constants.%F.specific_fn.72f]
// CHECK:STDOUT:   %bound_method.loc22_22: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %bound_method.loc22_22(%a.ref)
// CHECK:STDOUT:   %.loc22_22.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc22_22.2: ref %empty_struct_type = temporary %.loc22_22.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_23: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc22_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%T.loc3_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.loc7_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc7_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%Self) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.1
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%U.loc7_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%V) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.2
// CHECK:STDOUT:   %A => constants.%A.13025a.2
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1ceb36.2
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.848971.2
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.84bb3d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U.loc7_13.2 => constants.%V
// CHECK:STDOUT:   %U.patt.loc7_13.2 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2aef59.2
// CHECK:STDOUT:   %F => constants.%F.bb2dd4.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.955255.2
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.fef501.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc11_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc11_27.2 => constants.%A.13025a.2
// CHECK:STDOUT:   %I.type.loc11_35.2 => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfebb2.1
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.ab3b51.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.0fea45.1
// CHECK:STDOUT:   %F => constants.%F.d6ae34.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.%V.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%V.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%V.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.13025a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.2.%V) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%V, constants.%I.facet.5b0ece.1) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self => constants.%I.facet.5b0ece.1
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%A.13025a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.3
// CHECK:STDOUT:   %A => constants.%A.13025a.3
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1ceb36.3
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.848971.3
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.84bb3d.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc17_16.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc17_16.2 => constants.%W
// CHECK:STDOUT:   %A.loc17_32.2 => constants.%A.13025a.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@TestGeneric.%W.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U.loc7_13.2 => constants.%W
// CHECK:STDOUT:   %U.patt.loc7_13.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.2aef59.3
// CHECK:STDOUT:   %F => constants.%F.bb2dd4.3
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.955255.3
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.fef501.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%W) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%W
// CHECK:STDOUT:   %V.patt.loc11_14.2 => constants.%W
// CHECK:STDOUT:   %A.loc11_27.2 => constants.%A.13025a.3
// CHECK:STDOUT:   %I.type.loc11_35.2 => constants.%I.type.325e65.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfebb2.2
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.ab3b51.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.0fea45.2
// CHECK:STDOUT:   %F => constants.%F.d6ae34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%W) {
// CHECK:STDOUT:   %V => constants.%W
// CHECK:STDOUT:   %A => constants.%A.13025a.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_22 => constants.%require_complete.4aeca8.3
// CHECK:STDOUT:   %require_complete.loc12_12 => constants.%require_complete.5b8aee.2
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1ceb36.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%W.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@TestGeneric.%W.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(@TestGeneric.%W.loc17_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc3_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %A => constants.%A.235
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2af
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.91c
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %U.loc7_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %U.patt.loc7_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.885
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.684
// CHECK:STDOUT:   %F => constants.%F.a8d
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.67f
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.639
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc11_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc11_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc11_27.2 => constants.%A.235
// CHECK:STDOUT:   %I.type.loc11_35.2 => constants.%I.type.885
// CHECK:STDOUT:   %require_complete => constants.%complete_type.788
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness.f35
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.875
// CHECK:STDOUT:   %F => constants.%F.158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.235
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_22 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc12_12 => constants.%complete_type.0a6
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2af
// CHECK:STDOUT: }
// CHECK:STDOUT:
