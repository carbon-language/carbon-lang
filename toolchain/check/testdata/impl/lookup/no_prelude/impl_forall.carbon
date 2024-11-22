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
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = unbound_element_type %A.2, %T [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.4: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %A.3: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%V)> [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%V) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %I.type.3, %F.type.3 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.9: %I.type.2 = facet_value %A.3, %A.3 [symbolic]
// CHECK:STDOUT:   %.10: <witness> = interface_witness (%F.2) [symbolic]
// CHECK:STDOUT:   %.11: type = unbound_element_type %A.3, %V [symbolic]
// CHECK:STDOUT:   %.12: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %.13: <witness> = complete_type_witness %.12 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %A.4: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [template]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.15: type = unbound_element_type %A.4, %W [symbolic]
// CHECK:STDOUT:   %.16: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %.17: <witness> = complete_type_witness %.16 [symbolic]
// CHECK:STDOUT:   %I.type.4: type = facet_type <@I, @I(%W)> [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: type = assoc_entity_type %I.type.4, %F.type.4 [symbolic]
// CHECK:STDOUT:   %.20: %.19 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.2, @impl(%W) [symbolic]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.21: <witness> = interface_witness (%F.5) [symbolic]
// CHECK:STDOUT:   %.22: type = struct_type {} [template]
// CHECK:STDOUT:   %A.5: type = class_type @A, @A(%.22) [template]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [template]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.23: type = unbound_element_type %A.5, %.22 [template]
// CHECK:STDOUT:   %.24: type = struct_type {.n: %.22} [template]
// CHECK:STDOUT:   %.25: <witness> = complete_type_witness %.24 [template]
// CHECK:STDOUT:   %I.type.5: type = facet_type <@I, @I(%.22)> [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.1, @I(%.22) [template]
// CHECK:STDOUT:   %F.6: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.28: type = assoc_entity_type %I.type.5, %F.type.6 [template]
// CHECK:STDOUT:   %.29: %.28 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.2, @impl(%.22) [template]
// CHECK:STDOUT:   %F.7: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%F.7) [template]
// CHECK:STDOUT:   %struct: %.22 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .TestGeneric = %TestGeneric.decl
// CHECK:STDOUT:     .TestSpecific = %TestSpecific.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.1] {
// CHECK:STDOUT:     %T.patt.loc2_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc2_9.1, runtime_param<invalid> [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I] {
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
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %V.ref.loc10_26: type = name_ref V, %V.loc10_14.1 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:     %A.loc10_25.1: type = class_type @A, @A(constants.%V) [symbolic = %A.loc10_25.2 (constants.%A.3)]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %V.ref.loc10_34: type = name_ref V, %V.loc10_14.1 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc10_33.1: type = facet_type <@I, @I(constants.%V)> [symbolic = %I.type.loc10_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:     %V.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc10_14.1: type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [template = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt.loc16_16.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc16_16.1, runtime_param<invalid> [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %a.patt: @TestGeneric.%A.loc16_30.2 (%A.4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @TestGeneric.%A.loc16_30.2 (%A.4) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @TestGeneric.%W.loc16_16.2 (%W) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @TestGeneric.%W.loc16_16.2 (%W) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %W.ref.loc16_31: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %A.loc16_30.1: type = class_type @A, @A(constants.%W) [symbolic = %A.loc16_30.2 (constants.%A.4)]
// CHECK:STDOUT:     %W.ref.loc16_38: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc16_16.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.loc16_30.2 (%A.4) = value_param runtime_param0
// CHECK:STDOUT:     %a: @TestGeneric.%A.loc16_30.2 (%A.4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @TestGeneric.%W.loc16_16.2 (%W) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.loc16_16.2 (%W) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [template = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %A.5 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A.5 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.22 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.22 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %.loc20_23: %.22 = struct_literal ()
// CHECK:STDOUT:     %.loc20_21: type = converted %.loc20_23, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%.22) [template = constants.%A.5]
// CHECK:STDOUT:     %.loc20_31.1: %.22 = struct_literal ()
// CHECK:STDOUT:     %.loc20_31.2: type = converted %.loc20_31.1, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:     %a.param: %A.5 = value_param runtime_param0
// CHECK:STDOUT:     %a: %A.5 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.22 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.22 = return_slot %return.param
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
// CHECK:STDOUT:   %.loc7_26.2: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %.loc7_26.2 (constants.%.5)]
// CHECK:STDOUT:   %.loc7_26.3: @I.%.loc7_26.2 (%.5) = assoc_entity element0, %F.decl [symbolic = %.loc7_26.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%.loc7_14.1 (%.4) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%.loc7_14.1 (%.4) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.1.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%U (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_14.2: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc7_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.3: type = facet_access_type %Self.ref [symbolic = %.loc7_14.1 (constants.%.4)]
// CHECK:STDOUT:       %.loc7_14.4: type = converted %Self.ref, %.loc7_14.3 [symbolic = %.loc7_14.1 (constants.%.4)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc6_13.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.1.%.loc7_14.1 (%.4) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.1.%.loc7_14.1 (%.4) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.1.%U (%U) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.1.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_26.1: @I.%.loc7_26.2 (%.5) = assoc_entity element0, %F.decl [symbolic = %.loc7_26.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc7_26.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V.loc10_14.1: type) {
// CHECK:STDOUT:   %V.loc10_14.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc10_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc10_14.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc10_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %A.loc10_25.2: type = class_type @A, @A(%V.loc10_14.2) [symbolic = %A.loc10_25.2 (constants.%A.3)]
// CHECK:STDOUT:   %I.type.loc10_33.2: type = facet_type <@I, @I(%V.loc10_14.2)> [symbolic = %I.type.loc10_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V.loc10_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:   %.loc10_37.2: <witness> = interface_witness (%F) [symbolic = %.loc10_37.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc10_25.1 as %I.type.loc10_33.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {
// CHECK:STDOUT:       %self.patt: @F.2.%A (%A.3) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%A (%A.3) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.2.%V (%V) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%V (%V) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%A.loc10_25.1 [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:       %V.ref: type = name_ref V, @impl.%V.loc10_14.1 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %self.param: @F.2.%A (%A.3) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%A (%A.3) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%V (%V) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.2.%V (%V) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10_37.1: <witness> = interface_witness (%F.decl) [symbolic = %.loc10_37.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc10_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc2_9.1: type) {
// CHECK:STDOUT:   %T.loc2_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc2_9.2) [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:   %.loc3_8.2: type = unbound_element_type @A.%A (%A.2), @A.%T.loc2_9.2 (%T) [symbolic = %.loc3_8.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc4_1.2: type = struct_type {.n: @A.%T.loc2_9.2 (%T)} [symbolic = %.loc4_1.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc4_1.3: <witness> = complete_type_witness @A.%.loc4_1.2 (%.2) [symbolic = %.loc4_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2_9.1 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc3_8.1: @A.%.loc3_8.2 (%.1) = field_decl n, element0 [template]
// CHECK:STDOUT:     %.loc4_1.1: <witness> = complete_type_witness %.2 [symbolic = %.loc4_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2
// CHECK:STDOUT:     .n = %.loc3_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%U.loc6_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %.loc7_14.1: type = facet_access_type %Self [symbolic = %.loc7_14.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%.loc7_14.1 (%.4)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V.loc10_14.1: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_16.3: type = unbound_element_type @F.2.%A (%A.3), @F.2.%V (%V) [symbolic = %.loc12_16.3 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%A (%A.3)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.3) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @F.2.%.loc12_16.3 (%.11) = name_ref n, @A.%.loc3_8.1 [template = @A.%.loc3_8.1]
// CHECK:STDOUT:     %.loc12_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc12_16.2: @F.2.%V (%V) = bind_value %.loc12_16.1
// CHECK:STDOUT:     return %.loc12_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc16_16.1: type) {
// CHECK:STDOUT:   %W.loc16_16.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc16_16.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc16_16.2 (constants.%W.patt)]
// CHECK:STDOUT:   %A.loc16_30.2: type = class_type @A, @A(%W.loc16_16.2) [symbolic = %A.loc16_30.2 (constants.%A.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc17_14.2: type = facet_type <@I, @I(%W.loc16_16.2)> [symbolic = %I.type.loc17_14.2 (constants.%I.type.4)]
// CHECK:STDOUT:   %F.type.loc17_17: type = fn_type @F.1, @I(%W.loc16_16.2) [symbolic = %F.type.loc17_17 (constants.%F.type.4)]
// CHECK:STDOUT:   %.loc17_17.2: type = assoc_entity_type @TestGeneric.%I.type.loc17_14.2 (%I.type.4), @TestGeneric.%F.type.loc17_17 (%F.type.4) [symbolic = %.loc17_17.2 (constants.%.19)]
// CHECK:STDOUT:   %.loc17_17.3: @TestGeneric.%.loc17_17.2 (%.19) = assoc_entity element0, @I.%F.decl [symbolic = %.loc17_17.3 (constants.%.20)]
// CHECK:STDOUT:   %F.type.loc17_11: type = fn_type @F.2, @impl(%W.loc16_16.2) [symbolic = %F.type.loc17_11 (constants.%F.type.5)]
// CHECK:STDOUT:   %F: @TestGeneric.%F.type.loc17_11 (%F.type.5) = struct_value () [symbolic = %F (constants.%F.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.param_patt: type](%a.param_patt: @TestGeneric.%A.loc16_30.2 (%A.4)) -> @TestGeneric.%W.loc16_16.2 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.loc16_30.2 (%A.4) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %W.ref.loc17: type = name_ref W, %W.loc16_16.1 [symbolic = %W.loc16_16.2 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc17_14.1: type = facet_type <@I, @I(constants.%W)> [symbolic = %I.type.loc17_14.2 (constants.%I.type.4)]
// CHECK:STDOUT:     %.loc17_17.1: @TestGeneric.%.loc17_17.2 (%.19) = specific_constant @I.%.loc7_26.1, @I(constants.%W) [symbolic = %.loc17_17.3 (constants.%.20)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%.loc17_17.2 (%.19) = name_ref F, %.loc17_17.1 [symbolic = %.loc17_17.3 (constants.%.20)]
// CHECK:STDOUT:     %.loc17_11.1: @TestGeneric.%F.type.loc17_17 (%F.type.4) = interface_witness_access constants.%.21, element0 [symbolic = %F (constants.%F.5)]
// CHECK:STDOUT:     %.loc17_11.2: <bound method> = bound_method %a.ref, %.loc17_11.1
// CHECK:STDOUT:     %.loc17_11.3: <specific function> = specific_function %.loc17_11.2, @F.2(constants.%W)
// CHECK:STDOUT:     %F.call: init @TestGeneric.%W.loc16_16.2 (%W) = call %.loc17_11.3(%a.ref)
// CHECK:STDOUT:     %.loc17_22.1: @TestGeneric.%W.loc16_16.2 (%W) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc17_22.2: @TestGeneric.%W.loc16_16.2 (%W) = converted %F.call, %.loc17_22.1
// CHECK:STDOUT:     return %.loc17_22.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param_patt: %A.5) -> %.22 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.5 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:   %.loc21_16: %.22 = struct_literal ()
// CHECK:STDOUT:   %.loc21_14: type = converted %.loc21_16, constants.%.22 [template = constants.%.22]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(constants.%.22)> [template = constants.%I.type.5]
// CHECK:STDOUT:   %.loc21_18: %.28 = specific_constant @I.%.loc7_26.1, @I(constants.%.22) [template = constants.%.29]
// CHECK:STDOUT:   %F.ref: %.28 = name_ref F, %.loc21_18 [template = constants.%.29]
// CHECK:STDOUT:   %.loc21_11.1: %F.type.6 = interface_witness_access constants.%.30, element0 [template = constants.%F.7]
// CHECK:STDOUT:   %.loc21_11.2: <bound method> = bound_method %a.ref, %.loc21_11.1
// CHECK:STDOUT:   %.loc21_11.3: <specific function> = specific_function %.loc21_11.2, @F.2(constants.%.22)
// CHECK:STDOUT:   %F.call: init %.22 = call %.loc21_11.3(%a.ref)
// CHECK:STDOUT:   %.loc21_21.1: ref %.22 = temporary_storage
// CHECK:STDOUT:   %.loc21_21.2: ref %.22 = temporary %.loc21_21.1, %F.call
// CHECK:STDOUT:   %struct: %.22 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_23: %.22 = converted %F.call, %struct [template = constants.%struct]
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
// CHECK:STDOUT:   %.loc7_14.1 => constants.%.4
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
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.11
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.12
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.13
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
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.7
// CHECK:STDOUT:   %.loc7_26.3 => constants.%.8
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
// CHECK:STDOUT:   %A.loc10_25.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc10_33.2 => constants.%I.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
// CHECK:STDOUT:   %.loc10_37.2 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.2.%V) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%V, constants.%.9) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%.9
// CHECK:STDOUT:   %.loc7_14.1 => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%V.loc10_14.2) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc10_25.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc10_33.2 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.15
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.16
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.17
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
// CHECK:STDOUT:   %A.loc16_30.2 => constants.%A.4
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
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.19
// CHECK:STDOUT:   %.loc7_26.3 => constants.%.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%W) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%W
// CHECK:STDOUT:   %A.loc10_25.2 => constants.%A.4
// CHECK:STDOUT:   %I.type.loc10_33.2 => constants.%I.type.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.5
// CHECK:STDOUT:   %F => constants.%F.5
// CHECK:STDOUT:   %.loc10_37.2 => constants.%.21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%W) {
// CHECK:STDOUT:   %V => constants.%W
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_16.3 => constants.%.15
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
// CHECK:STDOUT:   %A.loc10_25.2 => constants.%A.4
// CHECK:STDOUT:   %I.type.loc10_33.2 => constants.%I.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%.22) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%.22
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%.22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.5
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.23
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.24
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.22) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%.22
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%.22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.6
// CHECK:STDOUT:   %F => constants.%F.6
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.28
// CHECK:STDOUT:   %.loc7_26.3 => constants.%.29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.22) {
// CHECK:STDOUT:   %V.loc10_14.2 => constants.%.22
// CHECK:STDOUT:   %V.patt.loc10_14.2 => constants.%.22
// CHECK:STDOUT:   %A.loc10_25.2 => constants.%A.5
// CHECK:STDOUT:   %I.type.loc10_33.2 => constants.%I.type.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.7
// CHECK:STDOUT:   %F => constants.%F.7
// CHECK:STDOUT:   %.loc10_37.2 => constants.%.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%.22) {
// CHECK:STDOUT:   %V => constants.%.22
// CHECK:STDOUT:   %A => constants.%A.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc12_16.3 => constants.%.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
