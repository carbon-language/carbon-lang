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

// CHECK:STDERR: impl_forall.carbon:[[@LINE+3]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl forall [V:! type] A(V) as I(V) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = interface_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %V.patt: type = symbolic_binding_pattern V, 0 [symbolic]
// CHECK:STDOUT:   %A.3: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = interface_type @I, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2, @impl(%V) [symbolic]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = unbound_element_type %A.3, %V [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type %.8 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %W.patt: type = symbolic_binding_pattern W, 0 [symbolic]
// CHECK:STDOUT:   %A.4: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [template]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.11: type = unbound_element_type %A.4, %W [symbolic]
// CHECK:STDOUT:   %.12: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %.13: <witness> = complete_type_witness %.12 [symbolic]
// CHECK:STDOUT:   %.14: type = ptr_type %.12 [symbolic]
// CHECK:STDOUT:   %I.type.4: type = interface_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %I.type.4, %F.type.3 [symbolic]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.2, @impl(%W) [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: type = struct_type {} [template]
// CHECK:STDOUT:   %A.5: type = class_type @A, @A(%.17) [template]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [template]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.18: type = unbound_element_type %A.5, %.17 [template]
// CHECK:STDOUT:   %.19: type = struct_type {.n: %.17} [template]
// CHECK:STDOUT:   %.20: <witness> = complete_type_witness %.19 [template]
// CHECK:STDOUT:   %.21: type = struct_type {.n: %.1} [template]
// CHECK:STDOUT:   %.22: type = ptr_type %.19 [template]
// CHECK:STDOUT:   %I.type.5: type = interface_type @I, @I(%.17) [template]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.1, @I(%.17) [template]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.23: type = assoc_entity_type %I.type.5, %F.type.5 [template]
// CHECK:STDOUT:   %.24: %.23 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.2, @impl(%.17) [template]
// CHECK:STDOUT:   %F.6: %F.type.6 = struct_value () [template]
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
// CHECK:STDOUT:     %V.patt.loc13_14.1: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_14.2 (constants.%V.patt)]
// CHECK:STDOUT:     %V.param_patt: type = value_param_pattern %V.patt.loc13_14.1, runtime_param<invalid> [symbolic = %V.patt.loc13_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %V.ref.loc13_26: type = name_ref V, %V.loc13_14.1 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:     %A.loc13_25.1: type = class_type @A, @A(constants.%V) [symbolic = %A.loc13_25.2 (constants.%A.3)]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %V.ref.loc13_34: type = name_ref V, %V.loc13_14.1 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc13_33.1: type = interface_type @I, @I(constants.%V) [symbolic = %I.type.loc13_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:     %V.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %V.loc13_14.1: type = bind_symbolic_name V, 0, %V.param [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [template = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt.loc19_16.1: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc19_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %W.param_patt: type = value_param_pattern %W.patt.loc19_16.1, runtime_param<invalid> [symbolic = %W.patt.loc19_16.2 (constants.%W.patt)]
// CHECK:STDOUT:     %a.patt: @TestGeneric.%A.loc19_30.2 (%A.4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @TestGeneric.%A.loc19_30.2 (%A.4) = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @TestGeneric.%W.loc19_16.2 (%W) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @TestGeneric.%W.loc19_16.2 (%W) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %W.ref.loc19_31: type = name_ref W, %W.loc19_16.1 [symbolic = %W.loc19_16.2 (constants.%W)]
// CHECK:STDOUT:     %A.loc19_30.1: type = class_type @A, @A(constants.%W) [symbolic = %A.loc19_30.2 (constants.%A.4)]
// CHECK:STDOUT:     %W.ref.loc19_38: type = name_ref W, %W.loc19_16.1 [symbolic = %W.loc19_16.2 (constants.%W)]
// CHECK:STDOUT:     %W.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %W.loc19_16.1: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.loc19_16.2 (constants.%W)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.loc19_30.2 (%A.4) = value_param runtime_param0
// CHECK:STDOUT:     %a: @TestGeneric.%A.loc19_30.2 (%A.4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @TestGeneric.%W.loc19_16.2 (%W) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.loc19_16.2 (%W) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [template = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %A.5 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A.5 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.17 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.17 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %.loc23_23: %.17 = struct_literal ()
// CHECK:STDOUT:     %.loc23_21: type = converted %.loc23_23, constants.%.17 [template = constants.%.17]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%.17) [template = constants.%A.5]
// CHECK:STDOUT:     %.loc23_31.1: %.17 = struct_literal ()
// CHECK:STDOUT:     %.loc23_31.2: type = converted %.loc23_31.1, constants.%.17 [template = constants.%.17]
// CHECK:STDOUT:     %a.param: %A.5 = value_param runtime_param0
// CHECK:STDOUT:     %a: %A.5 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.17 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.17 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc6_13.1: type) {
// CHECK:STDOUT:   %U.loc6_13.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc6_13.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_13.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_13.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%U.loc6_13.2) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%U.loc6_13.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.loc7_26.2: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %.loc7_26.2 (constants.%.5)]
// CHECK:STDOUT:   %.loc7_26.3: @I.%.loc7_26.2 (%.5) = assoc_entity element0, %F.decl [symbolic = %.loc7_26.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self (%Self) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.1.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%U (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_14.1: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc7_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.3: type = converted %Self.ref, %.loc7_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc6_13.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %self.param: @F.1.%Self (%Self) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.1.%Self (%Self) = bind_name self, %self.param
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
// CHECK:STDOUT: generic impl @impl(%V.loc13_14.1: type) {
// CHECK:STDOUT:   %V.loc13_14.2: type = bind_symbolic_name V, 0 [symbolic = %V.loc13_14.2 (constants.%V)]
// CHECK:STDOUT:   %V.patt.loc13_14.2: type = symbolic_binding_pattern V, 0 [symbolic = %V.patt.loc13_14.2 (constants.%V.patt)]
// CHECK:STDOUT:   %A.loc13_25.2: type = class_type @A, @A(%V.loc13_14.2) [symbolic = %A.loc13_25.2 (constants.%A.3)]
// CHECK:STDOUT:   %I.type.loc13_33.2: type = interface_type @I, @I(%V.loc13_14.2) [symbolic = %I.type.loc13_33.2 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%V.loc13_14.2) [symbolic = %F.type (constants.%F.type.2)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.2) = struct_value () [symbolic = %F (constants.%F.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc13_25.1 as %I.type.loc13_33.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.2) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.2)] {
// CHECK:STDOUT:       %self.patt: @F.2.%A (%A.3) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%A (%A.3) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.2.%V (%V) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.2.%V (%V) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%A.loc13_25.1 [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:       %V.ref: type = name_ref V, @impl.%V.loc13_14.1 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %self.param: @F.2.%A (%A.3) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.2.%A (%A.3) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.2.%V (%V) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.2.%V (%V) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc2_9.1: type) {
// CHECK:STDOUT:   %T.loc2_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc2_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc2_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc2_9.2) [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:   %.loc3_8.2: type = unbound_element_type @A.%A (%A.2), @A.%T.loc2_9.2 (%T) [symbolic = %.loc3_8.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc4_1.2: type = struct_type {.n: @A.%T.loc2_9.2 (%T)} [symbolic = %.loc4_1.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc4_1.3: <witness> = complete_type_witness @A.%.loc4_1.2 (%.3) [symbolic = %.loc4_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2_9.1 [symbolic = %T.loc2_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc3_8.1: @A.%.loc3_8.2 (%.2) = field_decl n, element0 [template]
// CHECK:STDOUT:     %.loc4_1.1: <witness> = complete_type_witness %.3 [symbolic = %.loc4_1.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2
// CHECK:STDOUT:     .n = %.loc3_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%U.loc6_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%U) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self (%Self)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V.loc13_14.1: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc15_16.3: type = unbound_element_type @F.2.%A (%A.3), @F.2.%V (%V) [symbolic = %.loc15_16.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%A (%A.3)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.3) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @F.2.%.loc15_16.3 (%.7) = name_ref n, @A.%.loc3_8.1 [template = @A.%.loc3_8.1]
// CHECK:STDOUT:     %.loc15_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc15_16.2: @F.2.%V (%V) = bind_value %.loc15_16.1
// CHECK:STDOUT:     return %.loc15_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc19_16.1: type) {
// CHECK:STDOUT:   %W.loc19_16.2: type = bind_symbolic_name W, 0 [symbolic = %W.loc19_16.2 (constants.%W)]
// CHECK:STDOUT:   %W.patt.loc19_16.2: type = symbolic_binding_pattern W, 0 [symbolic = %W.patt.loc19_16.2 (constants.%W.patt)]
// CHECK:STDOUT:   %A.loc19_30.2: type = class_type @A, @A(%W.loc19_16.2) [symbolic = %A.loc19_30.2 (constants.%A.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc20_14.2: type = interface_type @I, @I(%W.loc19_16.2) [symbolic = %I.type.loc20_14.2 (constants.%I.type.4)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%W.loc19_16.2) [symbolic = %F.type (constants.%F.type.3)]
// CHECK:STDOUT:   %.loc20_17.2: type = assoc_entity_type @TestGeneric.%I.type.loc20_14.2 (%I.type.4), @TestGeneric.%F.type (%F.type.3) [symbolic = %.loc20_17.2 (constants.%.15)]
// CHECK:STDOUT:   %.loc20_17.3: @TestGeneric.%.loc20_17.2 (%.15) = assoc_entity element0, @I.%F.decl [symbolic = %.loc20_17.3 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.param_patt: type](%a.param_patt: @TestGeneric.%A.loc19_30.2 (%A.4)) -> @TestGeneric.%W.loc19_16.2 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.loc19_30.2 (%A.4) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %W.ref.loc20: type = name_ref W, %W.loc19_16.1 [symbolic = %W.loc19_16.2 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc20_14.1: type = interface_type @I, @I(constants.%W) [symbolic = %I.type.loc20_14.2 (constants.%I.type.4)]
// CHECK:STDOUT:     %.loc20_17.1: @TestGeneric.%.loc20_17.2 (%.15) = specific_constant @I.%.loc7_26.1, @I(constants.%W) [symbolic = %.loc20_17.3 (constants.%.16)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%.loc20_17.2 (%.15) = name_ref F, %.loc20_17.1 [symbolic = %.loc20_17.3 (constants.%.16)]
// CHECK:STDOUT:     %.loc20_11.1: @TestGeneric.%F.type (%F.type.3) = interface_witness_access <error>, element0 [template = <error>]
// CHECK:STDOUT:     %.loc20_11.2: <bound method> = bound_method %a.ref, %.loc20_11.1
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a.param_patt: %A.5) -> %.17 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.5 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:   %.loc24_16: %.17 = struct_literal ()
// CHECK:STDOUT:   %.loc24_14: type = converted %.loc24_16, constants.%.17 [template = constants.%.17]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(constants.%.17) [template = constants.%I.type.5]
// CHECK:STDOUT:   %.loc24_18: %.23 = specific_constant @I.%.loc7_26.1, @I(constants.%.17) [template = constants.%.24]
// CHECK:STDOUT:   %F.ref: %.23 = name_ref F, %.loc24_18 [template = constants.%.24]
// CHECK:STDOUT:   %.loc24_11.1: %F.type.5 = interface_witness_access <error>, element0 [template = <error>]
// CHECK:STDOUT:   %.loc24_11.2: <bound method> = bound_method %a.ref, %.loc24_11.1
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@A.%T.loc2_9.2) {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%U.loc6_13.2) {
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
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.7
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.8
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%V
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.%V.loc13_14.2) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%V
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%V.loc13_14.2) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%V
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V.loc13_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc13_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc13_25.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc13_33.2 => constants.%I.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F => constants.%F.2
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
// CHECK:STDOUT: specific @impl(@impl.%V.loc13_14.2) {
// CHECK:STDOUT:   %V.loc13_14.2 => constants.%V
// CHECK:STDOUT:   %V.patt.loc13_14.2 => constants.%V
// CHECK:STDOUT:   %A.loc13_25.2 => constants.%A.3
// CHECK:STDOUT:   %I.type.loc13_33.2 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.11
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.12
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@TestGeneric.%W.loc19_16.2) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%W
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.loc19_16.2 => constants.%W
// CHECK:STDOUT:   %W.patt.loc19_16.2 => constants.%W
// CHECK:STDOUT:   %A.loc19_30.2 => constants.%A.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%W
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.15
// CHECK:STDOUT:   %.loc7_26.3 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%W) {
// CHECK:STDOUT:   %V.loc13_14.2 => constants.%W
// CHECK:STDOUT:   %V.patt.loc13_14.2 => constants.%W
// CHECK:STDOUT:   %A.loc13_25.2 => constants.%A.4
// CHECK:STDOUT:   %I.type.loc13_33.2 => constants.%I.type.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%W.loc19_16.2) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%W
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%.17) {
// CHECK:STDOUT:   %T.loc2_9.2 => constants.%.17
// CHECK:STDOUT:   %T.patt.loc2_9.2 => constants.%.17
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.5
// CHECK:STDOUT:   %.loc3_8.2 => constants.%.18
// CHECK:STDOUT:   %.loc4_1.2 => constants.%.19
// CHECK:STDOUT:   %.loc4_1.3 => constants.%.20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.17) {
// CHECK:STDOUT:   %U.loc6_13.2 => constants.%.17
// CHECK:STDOUT:   %U.patt.loc6_13.2 => constants.%.17
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5
// CHECK:STDOUT:   %F => constants.%F.5
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.23
// CHECK:STDOUT:   %.loc7_26.3 => constants.%.24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%.17) {
// CHECK:STDOUT:   %V.loc13_14.2 => constants.%.17
// CHECK:STDOUT:   %V.patt.loc13_14.2 => constants.%.17
// CHECK:STDOUT:   %A.loc13_25.2 => constants.%A.5
// CHECK:STDOUT:   %I.type.loc13_33.2 => constants.%I.type.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.6
// CHECK:STDOUT:   %F => constants.%F.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
