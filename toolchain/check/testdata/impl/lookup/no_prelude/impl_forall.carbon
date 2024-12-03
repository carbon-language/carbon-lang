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
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %A.elem.1: type = unbound_element_type %A.1, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n.1: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.n.1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%V)> [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%V) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %I.type.3, %F.type.3 [symbolic]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.2 = facet_value %A.2, %A.2 [symbolic]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %A.elem.2: type = unbound_element_type %A.2, %V [symbolic]
// CHECK:STDOUT:   %struct_type.n.2: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.n.2 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %A.3: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [template]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %A.elem.3: type = unbound_element_type %A.3, %W [symbolic]
// CHECK:STDOUT:   %struct_type.n.3: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.n.3 [symbolic]
// CHECK:STDOUT:   %I.type.4: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.3: type = assoc_entity_type %I.type.4, %F.type.4 [symbolic]
// CHECK:STDOUT:   %assoc0.3: %F.assoc_type.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.2, @impl(%W) [symbolic]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%F.5) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %A.4: type = class_type @A, @A(%empty_struct_type) [template]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [template]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %A.elem.4: type = unbound_element_type %A.4, %empty_struct_type [template]
// CHECK:STDOUT:   %struct_type.n.4: type = struct_type {.n: %empty_struct_type} [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %struct_type.n.4 [template]
// CHECK:STDOUT:   %I.type.5: type = facet_type <@I, @I(%empty_struct_type)> [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.1, @I(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.6: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.4: type = assoc_entity_type %I.type.5, %F.type.6 [template]
// CHECK:STDOUT:   %assoc0.4: %F.assoc_type.4 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.2, @impl(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.7: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %interface.3: <witness> = interface_witness (%F.7) [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .TestGeneric = %TestGeneric.decl
// CHECK:STDOUT:     .TestSpecific = %TestSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc2_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc2_9.1, runtime_param<invalid> [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %U.patt.loc6_13.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_13.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc6_13.1, runtime_param<invalid> [symbolic = %U.patt.loc6_13.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6_13.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc6_13.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %V.patt.loc10_14.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc10_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc10_14.1, runtime_param<invalid> [symbolic = %V.patt.loc10_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:     %V.ref.loc10_26: type = name_ref V, %V.loc10_14.1 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:     %A.loc10_27.1: type = class_type @A, @A(constants.%V) [symbolic = %A.loc10_27.2 (constants.%A.2)]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %V.ref.loc10_34: type = name_ref V, %V.loc10_14.1 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc10_35.1: type = facet_type <@I, @I(constants.%V)> [symbolic = %I.type.loc10_35.2 (constants.%I.type.3)]
// CHECK:STDOUT:     %V.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc10_14.1: type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [template = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt.loc16_16.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc16_16.1, runtime_param<invalid> [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %a.patt: @TestGeneric.%A.loc16_32.2 (%A.3) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @TestGeneric.%A.loc16_32.2 (%A.3) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @TestGeneric.%W.loc16_16.2 (%W) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @TestGeneric.%W.loc16_16.2 (%W) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:     %W.ref.loc16_31: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %A.loc16_32.1: type = class_type @A, @A(constants.%W) [symbolic = %A.loc16_32.2 (constants.%A.3)]
// CHECK:STDOUT:     %W.ref.loc16_38: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc16_16.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.loc16_32.2 (%A.3) = value_param runtime_param0
// CHECK:STDOUT:     %a: @TestGeneric.%A.loc16_32.2 (%A.3) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @TestGeneric.%W.loc16_16.2 (%W) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.loc16_16.2 (%W) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [template = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %A.4 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A.4 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:     %.loc20_23: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_24: type = converted %.loc20_23, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%empty_struct_type) [template = constants.%A.4]
// CHECK:STDOUT:     %.loc20_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_31.2: type = converted %.loc20_31.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %a.param: %A.4 = value_param runtime_param0
// CHECK:STDOUT:     %a: %A.4 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc6_13.1: type) {
// CHECK:STDOUT:   %U.loc6_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc6_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%U.loc6_13.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %F.assoc_type (constants.%F.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc7_26.2: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_26.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc7_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc7_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.1.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%U (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_14.1: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc7_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc7_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc7_14.2: type = converted %Self.ref, %Self.as_type.loc7_14.2 [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc6_13.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc7_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc7_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.1.%U (%U) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.1.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc7_26.1: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_26.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc7_26.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V.loc10_14.1: type) {
// CHECK:STDOUT:   %V.loc10_14.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc10_14.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc10_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %A.loc10_27.2: type = class_type @A, @A(%V.loc10_14.2) [symbolic = %A.loc10_27.2 (constants.%A.2)]
// CHECK:STDOUT:   %I.type.loc10_35.2: type = facet_type <@I, @I(%V.loc10_14.2)> [symbolic = %I.type.loc10_35.2 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V.loc10_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %interface.loc10_37.2: <witness> = interface_witness (%F) [symbolic = %interface.loc10_37.2 (constants.%interface.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc10_27.1 as %I.type.loc10_35.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {
// CHECK:STDOUT:       %self.patt: @F.2.%A (%A.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%A (%A.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.2.%V (%V) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%V (%V) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%A.loc10_27.1 [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:       %V.ref: type = name_ref V, @impl.%V.loc10_14.1 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %self.param: @F.2.%A (%A.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%A (%A.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%V (%V) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.2.%V (%V) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %interface.loc10_37.1: <witness> = interface_witness (%F.decl) [symbolic = %interface.loc10_37.2 (constants.%interface.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %interface.loc10_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc2_9.1: type) {
// CHECK:STDOUT:   %T.loc2_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc2_9.2) [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type @A.%A (%A.1), @A.%T.loc2_9.2 (%T) [symbolic = %A.elem (constants.%A.elem.1)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @A.%T.loc2_9.2 (%T)} [symbolic = %struct_type.n (constants.%struct_type.n.1)]
// CHECK:STDOUT:   %complete_type.loc4_1.2: <witness> = complete_type_witness @A.%struct_type.n (%struct_type.n.1) [symbolic = %complete_type.loc4_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2_9.1 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc3: @A.%A.elem (%A.elem.1) = field_decl n, element0 [template]
// CHECK:STDOUT:     %complete_type.loc4_1.1: <witness> = complete_type_witness %struct_type.n.1 [symbolic = %complete_type.loc4_1.2 (constants.%complete_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.1
// CHECK:STDOUT:     .n = %.loc3
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc4_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%U.loc6_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc7_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc7_14.1 (%Self.as_type)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V.loc10_14.1: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.elem: type = unbound_element_type @F.2.%A (%A.2), @F.2.%V (%V) [symbolic = %A.elem (constants.%A.elem.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%A (%A.2)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @F.2.%A.elem (%A.elem.2) = name_ref n, @A.%.loc3 [template = @A.%.loc3]
// CHECK:STDOUT:     %.loc12_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc12_16.2: @F.2.%V (%V) = bind_value %.loc12_16.1
// CHECK:STDOUT:     return %.loc12_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc16_16.1: type) {
// CHECK:STDOUT:   %W.loc16_16.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc16_16.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:   %A.loc16_32.2: type = class_type @A, @A(%W.loc16_16.2) [symbolic = %A.loc16_32.2 (constants.%A.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc17_16.2: type = facet_type <@I, @I(%W.loc16_16.2)> [symbolic = %I.type.loc17_16.2 (constants.%I.type.4)]
// CHECK:STDOUT:   %F.type.loc17_17: type = fn_type @F.1, @I(%W.loc16_16.2) [symbolic = %F.type.loc17_17 (constants.%F.type.4)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @TestGeneric.%I.type.loc17_16.2 (%I.type.4), @TestGeneric.%F.type.loc17_17 (%F.type.4) [symbolic = %F.assoc_type (constants.%F.assoc_type.3)]
// CHECK:STDOUT:   %assoc0: @TestGeneric.%F.assoc_type (%F.assoc_type.3) = assoc_entity element0, @I.%F.decl [symbolic = %assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:   %F.type.loc17_11: type = fn_type @F.2, @impl(%W.loc16_16.2) [symbolic = %F.type.loc17_11 (constants.%F.type.5)]
// CHECK:STDOUT:   %F: @TestGeneric.%F.type.loc17_11 (%F.type.5) = struct_value () [symbolic = %F (constants.%F.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.param_patt: type](%a.param_patt: @TestGeneric.%A.loc16_32.2 (%A.3)) -> @TestGeneric.%W.loc16_16.2 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.loc16_32.2 (%A.3) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %W.ref.loc17: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc17_16.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc17_16.2 (constants.%I.type.4)]
// CHECK:STDOUT:     %.loc17_17: @TestGeneric.%F.assoc_type (%F.assoc_type.3) = specific_constant @I.%assoc0.loc7_26.1, @I(constants.%W) [symbolic = %assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%F.assoc_type (%F.assoc_type.3) = name_ref F, %.loc17_17 [symbolic = %assoc0 (constants.%assoc0.3)]
// CHECK:STDOUT:     %impl.elem0: @TestGeneric.%F.type.loc17_17 (%F.type.4) = interface_witness_access constants.%interface.2, element0 [symbolic = %F (constants.%F.5)]
// CHECK:STDOUT:     %F.bound: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:     %F.specific_fn: <specific function> = specific_function %F.bound, @F.2(constants.%W)
// CHECK:STDOUT:     %F.call: init @TestGeneric.%W.loc16_16.2 (%W) = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:     %.loc17_22.1: @TestGeneric.%W.loc16_16.2 (%W) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc17_22.2: @TestGeneric.%W.loc16_16.2 (%W) = converted %F.call, %.loc17_22.1
// CHECK:STDOUT:     return %.loc17_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param_patt: %A.4) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.4 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:   %.loc21_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc21_17: type = converted %.loc21_16, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%empty_struct_type)> [template = constants.%I.type.5]
// CHECK:STDOUT:   %.loc21_18: %F.assoc_type.4 = specific_constant @I.%assoc0.loc7_26.1, @I(constants.%empty_struct_type) [template = constants.%assoc0.4]
// CHECK:STDOUT:   %F.ref: %F.assoc_type.4 = name_ref F, %.loc21_18 [template = constants.%assoc0.4]
// CHECK:STDOUT:   %impl.elem0: %F.type.6 = interface_witness_access constants.%interface.3, element0 [template = constants.%F.7]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref, %impl.elem0
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.bound, @F.2(constants.%empty_struct_type)
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   %.loc21_22.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc21_22.2: ref %empty_struct_type = temporary %.loc21_22.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_23: %empty_struct_type = converted %F.call, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc21_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%T.loc2_9.2) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%U) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%Self) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%U.loc6_13.2) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%V) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT:   %A.elem => constants.%A.elem.2
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.2
// CHECK:STDOUT:   %complete_type.loc4_1.2 => constants.%complete_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%V
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.2
// CHECK:STDOUT:   %assoc0.loc7_26.2 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.%V.loc10_14.2) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%V.loc10_14.2) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%V
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc10_27.2 => constants.%A.2
// CHECK:STDOUT:   %I.type.loc10_35.2 => constants.%I.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %interface.loc10_37.2 => constants.%interface.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.2.%V) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%V, constants.%I.facet) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc7_14.1 => constants.%A.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%V.loc10_14.2) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc10_27.2 => constants.%A.2
// CHECK:STDOUT:   %I.type.loc10_35.2 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:   %A.elem => constants.%A.elem.3
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.3
// CHECK:STDOUT:   %complete_type.loc4_1.2 => constants.%complete_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@TestGeneric.%W.loc16_16.2) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc16_16.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc16_16.2 => constants.%W
// CHECK:STDOUT:   %A.loc16_32.2 => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%W
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.3
// CHECK:STDOUT:   %assoc0.loc7_26.2 => constants.%assoc0.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%W) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %A.loc10_27.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc10_35.2 => constants.%I.type.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.5
// CHECK:STDOUT:   %F => constants.%F.5
// CHECK:STDOUT:   %interface.loc10_37.2 => constants.%interface.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%W) {
// CHECK:STDOUT:   %V => constants.%W
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.elem => constants.%A.elem.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%W.loc16_16.2) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%W
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(@TestGeneric.%W.loc16_16.2) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %A.loc10_27.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc10_35.2 => constants.%I.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:   %A.elem => constants.%A.elem.4
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.4
// CHECK:STDOUT:   %complete_type.loc4_1.2 => constants.%complete_type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.6
// CHECK:STDOUT:   %F => constants.%F.6
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.4
// CHECK:STDOUT:   %assoc0.loc7_26.2 => constants.%assoc0.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %A.loc10_27.2 => constants.%A.4
// CHECK:STDOUT:   %I.type.loc10_35.2 => constants.%I.type.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.7
// CHECK:STDOUT:   %F => constants.%F.7
// CHECK:STDOUT:   %interface.loc10_37.2 => constants.%interface.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%empty_struct_type) {
// CHECK:STDOUT:   %V => constants.%empty_struct_type
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.elem => constants.%A.elem.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
