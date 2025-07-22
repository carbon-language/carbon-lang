// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/impl_forall.carbon

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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.13025a.1: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.1: type = unbound_element_type %A.13025a.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.1: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.1: <witness> = complete_type_witness %struct_type.n.848971.1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325e65.1: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self.209206.1: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.928: type = facet_access_type %Self.209206.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4be: type = pattern_type %Self.as_type.928 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %I.F.type.2aef59.1: type = fn_type @I.F, @I(%U) [symbolic]
// CHECK:STDOUT:   %I.F.bb2dd4.1: %I.F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.1: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.1: %I.assoc_type.1e5078.1 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %A.13025a.2: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.325e65.2: type = facet_type <@I, @I(%V)> [symbolic]
// CHECK:STDOUT:   %Self.209206.2: %I.type.325e65.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.2aef59.2: type = fn_type @I.F, @I(%V) [symbolic]
// CHECK:STDOUT:   %I.F.bb2dd4.2: %I.F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.2: type = assoc_entity_type @I, @I(%V) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.2: %I.assoc_type.1e5078.2 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfebb2.1: <witness> = require_complete_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.40ab31.1: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%V) [symbolic]
// CHECK:STDOUT:   %pattern_type.68d8d3.1: type = pattern_type %A.13025a.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %V [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.0fea45.1: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.d6ae34.1: %A.as.I.impl.F.type.0fea45.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet.de3: %I.type.325e65.2 = facet_value %A.13025a.2, (%I.impl_witness.40ab31.1) [symbolic]
// CHECK:STDOUT:   %require_complete.4aeca8.2: <witness> = require_complete_type %V [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.2: type = unbound_element_type %A.13025a.2, %V [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.2: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.2: <witness> = complete_type_witness %struct_type.n.848971.2 [symbolic]
// CHECK:STDOUT:   %require_complete.5b8aee.1: <witness> = require_complete_type %A.13025a.2 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %A.13025a.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %pattern_type.68d8d3.2: type = pattern_type %A.13025a.3 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.3: type = pattern_type %W [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [concrete]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4aeca8.3: <witness> = require_complete_type %W [symbolic]
// CHECK:STDOUT:   %A.elem.1ceb36.3: type = unbound_element_type %A.13025a.3, %W [symbolic]
// CHECK:STDOUT:   %struct_type.n.848971.3: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %complete_type.84bb3d.3: <witness> = complete_type_witness %struct_type.n.848971.3 [symbolic]
// CHECK:STDOUT:   %require_complete.5b8aee.2: <witness> = require_complete_type %A.13025a.3 [symbolic]
// CHECK:STDOUT:   %I.type.325e65.3: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %Self.209206.3: %I.type.325e65.3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.2aef59.3: type = fn_type @I.F, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.F.bb2dd4.3: %I.F.type.2aef59.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.3: type = assoc_entity_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.3: %I.assoc_type.1e5078.3 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfebb2.2: <witness> = require_complete_type %I.type.325e65.3 [symbolic]
// CHECK:STDOUT:   %I.impl_witness.40ab31.2: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.type.0fea45.2: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%W) [symbolic]
// CHECK:STDOUT:   %A.as.I.impl.F.d6ae34.2: %A.as.I.impl.F.type.0fea45.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.13025a.3, @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %I.facet.6d2: %I.type.325e65.3 = facet_value %A.13025a.3, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.fe1: type = fn_type_with_self_type %I.F.type.2aef59.3, %I.facet.6d2 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.fe1 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.F(%W, %I.facet.6d2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A.235: type = class_type @A, @A(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.496: type = pattern_type %A.235 [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [concrete]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A.elem.2af: type = unbound_element_type %A.235, %empty_struct_type [concrete]
// CHECK:STDOUT:   %struct_type.n.91c: type = struct_type {.n: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.n.91c [concrete]
// CHECK:STDOUT:   %I.type.885: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.4e2: %I.type.885 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.684: type = fn_type @I.F, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.F.a8d: %I.F.type.684 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.22c: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.722: %I.assoc_type.22c = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.788: <witness> = complete_type_witness %I.type.885 [concrete]
// CHECK:STDOUT:   %I.impl_witness.0f2: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.type.875: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.158: %A.as.I.impl.F.type.875 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.bd5: %I.type.885 = facet_value %A.235, (%I.impl_witness.0f2) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %I.F.type.684, %I.facet.bd5 [concrete]
// CHECK:STDOUT:   %A.as.I.impl.F.specific_fn: <specific function> = specific_function %A.as.I.impl.F.158, @A.as.I.impl.F(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.069: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d5a: %T.as.Destroy.impl.Op.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.d5a, @T.as.Destroy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .TestGeneric = %TestGeneric.decl
// CHECK:STDOUT:     .TestSpecific = %TestSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc3_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc7_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @A.as.I.impl [concrete] {
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc11_26: type = name_ref V, %V.loc11_14.2 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:     %A.loc11_27.2: type = class_type @A, @A(constants.%V) [symbolic = %A.loc11_27.1 (constants.%A.13025a.2)]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc11_34: type = name_ref V, %V.loc11_14.2 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc11_35.2: type = facet_type <@I, @I(constants.%V)> [symbolic = %I.type.loc11_35.1 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     %V.loc11_14.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@A.as.I.impl.%A.as.I.impl.F.decl), @A.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @A.as.I.impl(constants.%V) [symbolic = @A.as.I.impl.%I.impl_witness (constants.%I.impl_witness.40ab31.1)]
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [concrete = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt: %pattern_type.98f = symbolic_binding_pattern W, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @TestGeneric.%pattern_type.loc17_26 (%pattern_type.68d8d3.2) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @TestGeneric.%pattern_type.loc17_26 (%pattern_type.68d8d3.2) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @TestGeneric.%pattern_type.loc17_35 (%pattern_type.7dcd0a.3) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @TestGeneric.%pattern_type.loc17_35 (%pattern_type.7dcd0a.3) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.ref.loc17_38: type = name_ref W, %W.loc17_16.2 [symbolic = %W.loc17_16.1 (constants.%W)]
// CHECK:STDOUT:     %W.loc17_16.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc17_16.1 (constants.%W)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.loc17_32.1 (%A.13025a.3) = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %A.loc17_32.2 [symbolic = %A.loc17_32.1 (constants.%A.13025a.3)] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %W.ref.loc17_31: type = name_ref W, %W.loc17_16.2 [symbolic = %W.loc17_16.1 (constants.%W)]
// CHECK:STDOUT:       %A.loc17_32.2: type = class_type @A, @A(constants.%W) [symbolic = %A.loc17_32.1 (constants.%A.13025a.3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @TestGeneric.%A.loc17_32.1 (%A.13025a.3) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @TestGeneric.%W.loc17_16.1 (%W) = out_param call_param1
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.loc17_16.1 (%W) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [concrete = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %pattern_type.496 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.496 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: generic interface @I(%U.loc7_13.2: type) {
// CHECK:STDOUT:   %U.loc7_13.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc7_13.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc7_13.1)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%U.loc7_13.1) [symbolic = %I.F.type (constants.%I.F.type.2aef59.1)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.2aef59.1) = struct_value () [symbolic = %I.F (constants.%I.F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%U.loc7_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5078.1)]
// CHECK:STDOUT:   %assoc0.loc8_26.2: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc8_26.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:     %I.F.decl: @I.%I.F.type (%I.F.type.2aef59.1) = fn_decl @I.F [symbolic = @I.%I.F (constants.%I.F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @I.F.%pattern_type.loc8_8 (%pattern_type.4be) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @I.F.%pattern_type.loc8_8 (%pattern_type.4be) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @I.F.%pattern_type.loc8_22 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @I.F.%pattern_type.loc8_22 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc7_13.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @I.F.%Self.as_type.loc8_14.1 (%Self.as_type.928) = value_param call_param0
// CHECK:STDOUT:       %.loc8_14.1: type = splice_block %.loc8_14.3 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.928)] {
// CHECK:STDOUT:         %.loc8_14.2: @I.F.%I.type (%I.type.325e65.1) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.ref: @I.F.%I.type (%I.type.325e65.1) = name_ref Self, %.loc8_14.2 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.as_type.loc8_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:         %.loc8_14.3: type = converted %Self.ref, %Self.as_type.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.F.%Self.as_type.loc8_14.1 (%Self.as_type.928) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @I.F.%U (%U) = out_param call_param1
// CHECK:STDOUT:       %return: ref @I.F.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_26.1: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc8_26.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc8_26.1
// CHECK:STDOUT:     witness = (%I.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.I.impl(%V.loc11_14.2: type) {
// CHECK:STDOUT:   %V.loc11_14.1: type = bind_symbolic_name V, 0 [symbolic = %V.loc11_14.1 (constants.%V)]
// CHECK:STDOUT:   %A.loc11_27.1: type = class_type @A, @A(%V.loc11_14.1) [symbolic = %A.loc11_27.1 (constants.%A.13025a.2)]
// CHECK:STDOUT:   %I.type.loc11_35.1: type = facet_type <@I, @I(%V.loc11_14.1)> [symbolic = %I.type.loc11_35.1 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc11_35.1 [symbolic = %require_complete (constants.%require_complete.cfebb2.1)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table, @A.as.I.impl(%V.loc11_14.1) [symbolic = %I.impl_witness (constants.%I.impl_witness.40ab31.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type: type = fn_type @A.as.I.impl.F, @A.as.I.impl(%V.loc11_14.1) [symbolic = %A.as.I.impl.F.type (constants.%A.as.I.impl.F.type.0fea45.1)]
// CHECK:STDOUT:   %A.as.I.impl.F: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.0fea45.1) = struct_value () [symbolic = %A.as.I.impl.F (constants.%A.as.I.impl.F.d6ae34.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc11_27.2 as %I.type.loc11_35.2 {
// CHECK:STDOUT:     %A.as.I.impl.F.decl: @A.as.I.impl.%A.as.I.impl.F.type (%A.as.I.impl.F.type.0fea45.1) = fn_decl @A.as.I.impl.F [symbolic = @A.as.I.impl.%A.as.I.impl.F (constants.%A.as.I.impl.F.d6ae34.1)] {
// CHECK:STDOUT:       %self.patt: @A.as.I.impl.F.%pattern_type.loc12_8 (%pattern_type.68d8d3.1) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.I.impl.F.%pattern_type.loc12_8 (%pattern_type.68d8d3.1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @A.as.I.impl.F.%pattern_type.loc12_22 (%pattern_type.7dcd0a.2) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @A.as.I.impl.F.%pattern_type.loc12_22 (%pattern_type.7dcd0a.2) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %V.ref: type = name_ref V, @A.as.I.impl.%V.loc11_14.2 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %self.param: @A.as.I.impl.F.%A (%A.13025a.2) = value_param call_param0
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @A.as.I.impl.%A.loc11_27.2 [symbolic = %A (constants.%A.13025a.2)]
// CHECK:STDOUT:       %self: @A.as.I.impl.F.%A (%A.13025a.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @A.as.I.impl.F.%V (%V) = out_param call_param1
// CHECK:STDOUT:       %return: ref @A.as.I.impl.F.%V (%V) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .V = <poisoned>
// CHECK:STDOUT:     .F = %A.as.I.impl.F.decl
// CHECK:STDOUT:     witness = file.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc3_9.2: type) {
// CHECK:STDOUT:   %T.loc3_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc3_9.1 [symbolic = %require_complete (constants.%require_complete.4aeca8.1)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc3_9.1) [symbolic = %A (constants.%A.13025a.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %T.loc3_9.1 [symbolic = %A.elem (constants.%A.elem.1ceb36.1)]
// CHECK:STDOUT:   %struct_type.n.loc5_1.2: type = struct_type {.n: @A.%T.loc3_9.1 (%T)} [symbolic = %struct_type.n.loc5_1.2 (constants.%struct_type.n.848971.1)]
// CHECK:STDOUT:   %complete_type.loc5_1.2: <witness> = complete_type_witness %struct_type.n.loc5_1.2 [symbolic = %complete_type.loc5_1.2 (constants.%complete_type.84bb3d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc3_9.2 [symbolic = %T.loc3_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4: @A.%A.elem (%A.elem.1ceb36.1) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %struct_type.n.loc5_1.1: type = struct_type {.n: %T} [symbolic = %struct_type.n.loc5_1.2 (constants.%struct_type.n.848971.1)]
// CHECK:STDOUT:     %complete_type.loc5_1.1: <witness> = complete_type_witness %struct_type.n.loc5_1.1 [symbolic = %complete_type.loc5_1.2 (constants.%complete_type.84bb3d.1)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc5_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.13025a.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%U.loc7_13.2: type, @I.%Self.1: @I.%I.type (%I.type.325e65.1)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:   %Self.as_type.loc8_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:   %pattern_type.loc8_8: type = pattern_type %Self.as_type.loc8_14.1 [symbolic = %pattern_type.loc8_8 (constants.%pattern_type.4be)]
// CHECK:STDOUT:   %pattern_type.loc8_22: type = pattern_type %U [symbolic = %pattern_type.loc8_22 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.as_type.loc8_14.1 (%Self.as_type.928)) -> @I.F.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.I.impl.F(@A.as.I.impl.%V.loc11_14.2: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.13025a.2)]
// CHECK:STDOUT:   %pattern_type.loc12_8: type = pattern_type %A [symbolic = %pattern_type.loc12_8 (constants.%pattern_type.68d8d3.1)]
// CHECK:STDOUT:   %pattern_type.loc12_22: type = pattern_type %V [symbolic = %pattern_type.loc12_22 (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_22: <witness> = require_complete_type %V [symbolic = %require_complete.loc12_22 (constants.%require_complete.4aeca8.2)]
// CHECK:STDOUT:   %require_complete.loc12_12: <witness> = require_complete_type %A [symbolic = %require_complete.loc12_12 (constants.%require_complete.5b8aee.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %V [symbolic = %A.elem (constants.%A.elem.1ceb36.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.as.I.impl.F.%A (%A.13025a.2)) -> @A.as.I.impl.F.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @A.as.I.impl.F.%A (%A.13025a.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @A.as.I.impl.F.%A.elem (%A.elem.1ceb36.2) = name_ref n, @A.%.loc4 [concrete = @A.%.loc4]
// CHECK:STDOUT:     %.loc13_16.1: ref @A.as.I.impl.F.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc13_16.2: @A.as.I.impl.F.%V (%V) = bind_value %.loc13_16.1
// CHECK:STDOUT:     return %.loc13_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc17_16.2: type) {
// CHECK:STDOUT:   %W.loc17_16.1: type = bind_symbolic_name W, 0 [symbolic = %W.loc17_16.1 (constants.%W)]
// CHECK:STDOUT:   %A.loc17_32.1: type = class_type @A, @A(%W.loc17_16.1) [symbolic = %A.loc17_32.1 (constants.%A.13025a.3)]
// CHECK:STDOUT:   %pattern_type.loc17_26: type = pattern_type %A.loc17_32.1 [symbolic = %pattern_type.loc17_26 (constants.%pattern_type.68d8d3.2)]
// CHECK:STDOUT:   %pattern_type.loc17_35: type = pattern_type %W.loc17_16.1 [symbolic = %pattern_type.loc17_35 (constants.%pattern_type.7dcd0a.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17_35: <witness> = require_complete_type %W.loc17_16.1 [symbolic = %require_complete.loc17_35 (constants.%require_complete.4aeca8.3)]
// CHECK:STDOUT:   %require_complete.loc17_27: <witness> = require_complete_type %A.loc17_32.1 [symbolic = %require_complete.loc17_27 (constants.%require_complete.5b8aee.2)]
// CHECK:STDOUT:   %I.type.loc18_16.2: type = facet_type <@I, @I(%W.loc17_16.1)> [symbolic = %I.type.loc18_16.2 (constants.%I.type.325e65.3)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type %I.type.loc18_16.2 [symbolic = %require_complete.loc18 (constants.%require_complete.cfebb2.2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%W.loc17_16.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5078.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%I.assoc_type (%I.assoc_type.1e5078.3) = assoc_entity element0, @I.%I.F.decl [symbolic = %assoc0 (constants.%assoc0.8f0422.3)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %A.loc17_32.1, @I, @I(%W.loc17_16.1) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%W.loc17_16.1) [symbolic = %I.F.type (constants.%I.F.type.2aef59.3)]
// CHECK:STDOUT:   %I.facet: @TestGeneric.%I.type.loc18_16.2 (%I.type.325e65.3) = facet_value %A.loc17_32.1, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet.6d2)]
// CHECK:STDOUT:   %.loc18_11: type = fn_type_with_self_type %I.F.type, %I.facet [symbolic = %.loc18_11 (constants.%.fe1)]
// CHECK:STDOUT:   %impl.elem0.loc18_11.2: @TestGeneric.%.loc18_11 (%.fe1) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_11.2: <specific function> = specific_impl_function %impl.elem0.loc18_11.2, @I.F(%W.loc17_16.1, %I.facet) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @TestGeneric.%A.loc17_32.1 (%A.13025a.3)) -> @TestGeneric.%W.loc17_16.1 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.loc17_32.1 (%A.13025a.3) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc18: type = name_ref W, %W.loc17_16.2 [symbolic = %W.loc17_16.1 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc18_16.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc18_16.2 (constants.%I.type.325e65.3)]
// CHECK:STDOUT:     %.loc18_17: @TestGeneric.%I.assoc_type (%I.assoc_type.1e5078.3) = specific_constant @I.%assoc0.loc8_26.1, @I(constants.%W) [symbolic = %assoc0 (constants.%assoc0.8f0422.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%I.assoc_type (%I.assoc_type.1e5078.3) = name_ref F, %.loc18_17 [symbolic = %assoc0 (constants.%assoc0.8f0422.3)]
// CHECK:STDOUT:     %impl.elem0.loc18_11.1: @TestGeneric.%.loc18_11 (%.fe1) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc18_11: <bound method> = bound_method %a.ref, %impl.elem0.loc18_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc18_11.1: <specific function> = specific_impl_function %impl.elem0.loc18_11.1, @I.F(constants.%W, constants.%I.facet.6d2) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc18_21: <bound method> = bound_method %a.ref, %specific_impl_fn.loc18_11.1
// CHECK:STDOUT:     %.loc18_21: init @TestGeneric.%W.loc17_16.1 (%W) = call %bound_method.loc18_21(%a.ref)
// CHECK:STDOUT:     %.loc18_22.1: @TestGeneric.%W.loc17_16.1 (%W) = value_of_initializer %.loc18_21
// CHECK:STDOUT:     %.loc18_22.2: @TestGeneric.%W.loc17_16.1 (%W) = converted %.loc18_21, %.loc18_22.1
// CHECK:STDOUT:     return %.loc18_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param: %A.235) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.235 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:   %.loc22_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_17: type = converted %.loc22_16, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [concrete = constants.%I.type.885]
// CHECK:STDOUT:   %.loc22_18: %I.assoc_type.22c = specific_constant @I.%assoc0.loc8_26.1, @I(constants.%empty_struct_type) [concrete = constants.%assoc0.722]
// CHECK:STDOUT:   %F.ref: %I.assoc_type.22c = name_ref F, %.loc22_18 [concrete = constants.%assoc0.722]
// CHECK:STDOUT:   %impl.elem0: %.8fd = impl_witness_access constants.%I.impl_witness.0f2, element0 [concrete = constants.%A.as.I.impl.F.158]
// CHECK:STDOUT:   %bound_method.loc22_11: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @A.as.I.impl.F(constants.%empty_struct_type) [concrete = constants.%A.as.I.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_22.1: <bound method> = bound_method %a.ref, %specific_fn
// CHECK:STDOUT:   %A.as.I.impl.F.call: init %empty_struct_type = call %bound_method.loc22_22.1(%a.ref)
// CHECK:STDOUT:   %.loc22_22.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc22_22.2: ref %empty_struct_type = temporary %.loc22_22.1, %A.as.I.impl.F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc22_23: %empty_struct_type = converted %A.as.I.impl.F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc22_22.1, constants.%T.as.Destroy.impl.Op.d5a
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.d5a, @T.as.Destroy.impl.Op(constants.%empty_struct_type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_22.2: <bound method> = bound_method %.loc22_22.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c28 = addr_of %.loc22_22.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22_22.2(%addr)
// CHECK:STDOUT:   return %.loc22_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%U, constants.%Self.209206.1) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.1
// CHECK:STDOUT:   %Self => constants.%Self.209206.1
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%Self.as_type.928
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.4be
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%V) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.2
// CHECK:STDOUT:   %A => constants.%A.13025a.2
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1ceb36.2
// CHECK:STDOUT:   %struct_type.n.loc5_1.2 => constants.%struct_type.n.848971.2
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.84bb3d.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.209206.2
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.2aef59.2
// CHECK:STDOUT:   %I.F => constants.%I.F.bb2dd4.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e5078.2
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.8f0422.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%V) {
// CHECK:STDOUT:   %V.loc11_14.1 => constants.%V
// CHECK:STDOUT:   %A.loc11_27.1 => constants.%A.13025a.2
// CHECK:STDOUT:   %I.type.loc11_35.1 => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfebb2.1
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.40ab31.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.0fea45.1
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.d6ae34.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.13025a.2
// CHECK:STDOUT:   %pattern_type.loc12_8 => constants.%pattern_type.68d8d3.1
// CHECK:STDOUT:   %pattern_type.loc12_22 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%V, constants.%I.facet.de3) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self => constants.%I.facet.de3
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%A.13025a.2
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.68d8d3.1
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4aeca8.3
// CHECK:STDOUT:   %A => constants.%A.13025a.3
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1ceb36.3
// CHECK:STDOUT:   %struct_type.n.loc5_1.2 => constants.%struct_type.n.848971.3
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.84bb3d.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc17_16.1 => constants.%W
// CHECK:STDOUT:   %A.loc17_32.1 => constants.%A.13025a.3
// CHECK:STDOUT:   %pattern_type.loc17_26 => constants.%pattern_type.68d8d3.2
// CHECK:STDOUT:   %pattern_type.loc17_35 => constants.%pattern_type.7dcd0a.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.209206.3
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.2aef59.3
// CHECK:STDOUT:   %I.F => constants.%I.F.bb2dd4.3
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e5078.3
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.8f0422.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%W) {
// CHECK:STDOUT:   %V.loc11_14.1 => constants.%W
// CHECK:STDOUT:   %A.loc11_27.1 => constants.%A.13025a.3
// CHECK:STDOUT:   %I.type.loc11_35.1 => constants.%I.type.325e65.3
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfebb2.2
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.40ab31.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.0fea45.2
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.d6ae34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%W, constants.%I.facet.6d2) {
// CHECK:STDOUT:   %U => constants.%W
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.3
// CHECK:STDOUT:   %Self => constants.%I.facet.6d2
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%A.13025a.3
// CHECK:STDOUT:   %pattern_type.loc8_8 => constants.%pattern_type.68d8d3.2
// CHECK:STDOUT:   %pattern_type.loc8_22 => constants.%pattern_type.7dcd0a.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc3_9.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %A => constants.%A.235
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2af
// CHECK:STDOUT:   %struct_type.n.loc5_1.2 => constants.%struct_type.n.91c
// CHECK:STDOUT:   %complete_type.loc5_1.2 => constants.%complete_type.0a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %U.loc7_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.885
// CHECK:STDOUT:   %Self.2 => constants.%Self.4e2
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.684
// CHECK:STDOUT:   %I.F => constants.%I.F.a8d
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.22c
// CHECK:STDOUT:   %assoc0.loc8_26.2 => constants.%assoc0.722
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc11_14.1 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc11_27.1 => constants.%A.235
// CHECK:STDOUT:   %I.type.loc11_35.1 => constants.%I.type.885
// CHECK:STDOUT:   %require_complete => constants.%complete_type.788
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness.0f2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.I.impl.F.type => constants.%A.as.I.impl.F.type.875
// CHECK:STDOUT:   %A.as.I.impl.F => constants.%A.as.I.impl.F.158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.I.impl.F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.235
// CHECK:STDOUT:   %pattern_type.loc12_8 => constants.%pattern_type.496
// CHECK:STDOUT:   %pattern_type.loc12_22 => constants.%pattern_type.a96
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_22 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc12_12 => constants.%complete_type.0a6
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2af
// CHECK:STDOUT: }
// CHECK:STDOUT:
