// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/no_prelude/impl_forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/no_prelude/impl_forall.carbon

// --- fail_todo_impl_forall.carbon

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
  // CHECK:STDERR: fail_todo_impl_forall.carbon:[[@LINE+4]]:10: error: cannot access member of interface `I` in type `A` that does not implement that interface
  // CHECK:STDERR:   return a.(I(W).F)();
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR:
  return a.(I(W).F)();
}

fn TestSpecific(a: A({})) -> {} {
  // CHECK:STDERR: fail_todo_impl_forall.carbon:[[@LINE+3]]:10: error: cannot access member of interface `I` in type `A` that does not implement that interface
  // CHECK:STDERR:   return a.(I({}).F)();
  // CHECK:STDERR:          ^~~~~~~~~~~
  return a.(I({}).F)();
}

// CHECK:STDOUT: --- fail_todo_impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = interface_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic]
// CHECK:STDOUT:   %A.3: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %I.type.3: type = interface_type @I, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %I.type.3, %F.type.3 [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type %A.3, %V [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %.12: <witness> = complete_type_witness %.11 [symbolic]
// CHECK:STDOUT:   %.13: type = ptr_type %.11 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W, 0 [symbolic]
// CHECK:STDOUT:   %A.4: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [template]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.14: type = unbound_element_type %A.4, %W [symbolic]
// CHECK:STDOUT:   %.15: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %.16: <witness> = complete_type_witness %.15 [symbolic]
// CHECK:STDOUT:   %.17: type = ptr_type %.15 [symbolic]
// CHECK:STDOUT:   %I.type.4: type = interface_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %I.type.4, %F.type.4 [symbolic]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.20: type = struct_type {} [template]
// CHECK:STDOUT:   %A.5: type = class_type @A, @A(%.20) [template]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [template]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.21: type = unbound_element_type %A.5, %.20 [template]
// CHECK:STDOUT:   %.22: type = struct_type {.n: %.20} [template]
// CHECK:STDOUT:   %.23: <witness> = complete_type_witness %.22 [template]
// CHECK:STDOUT:   %.24: type = struct_type {.n: %.1} [template]
// CHECK:STDOUT:   %.25: type = ptr_type %.22 [template]
// CHECK:STDOUT:   %I.type.5: type = interface_type @I, @I(%.20) [template]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.1, @I(%.20) [template]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.26: type = assoc_entity_type %I.type.5, %F.type.5 [template]
// CHECK:STDOUT:   %.27: %.26 = assoc_entity element0, @I.%F.decl [template]
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {
// CHECK:STDOUT:     %V.patt: type = symbolic_binding_pattern V, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %V.param: type = param V, runtime_param<invalid>
// CHECK:STDOUT:     %V.loc10: type = bind_symbolic_name V, 0, %V.param [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %V.ref.loc10_26: type = name_ref V, %V.loc10 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:     %A.loc10: type = class_type @A, @A(constants.%V) [symbolic = %A.1 (constants.%A.3)]
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %V.ref.loc10_34: type = name_ref V, %V.loc10 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:     %I.type.loc10: type = interface_type @I, @I(constants.%V) [symbolic = %I.type.1 (constants.%I.type.3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [template = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.patt: type = symbolic_binding_pattern W, 0
// CHECK:STDOUT:     %a.patt: @TestGeneric.%A.1 (%A.4) = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %W.param: type = param W, runtime_param<invalid>
// CHECK:STDOUT:     %W.loc16: type = bind_symbolic_name W, 0, %W.param [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %W.ref.loc16_31: type = name_ref W, %W.loc16 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:     %A.loc16: type = class_type @A, @A(constants.%W) [symbolic = %A.1 (constants.%A.4)]
// CHECK:STDOUT:     %a.param: @TestGeneric.%A.1 (%A.4) = param a, runtime_param0
// CHECK:STDOUT:     %a: @TestGeneric.%A.1 (%A.4) = bind_name a, %a.param
// CHECK:STDOUT:     %W.ref.loc16_38: type = name_ref W, %W.loc16 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:     %return: ref @TestGeneric.%W.1 (%W) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [template = constants.%TestSpecific] {
// CHECK:STDOUT:     %a.patt: %A.5 = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %.loc24_23: %.20 = struct_literal ()
// CHECK:STDOUT:     %.loc24_21: type = converted %.loc24_23, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:     %A: type = class_type @A, @A(constants.%.20) [template = constants.%A.5]
// CHECK:STDOUT:     %a.param: %A.5 = param a, runtime_param0
// CHECK:STDOUT:     %a: %A.5 = bind_name a, %a.param
// CHECK:STDOUT:     %.loc24_31.1: %.20 = struct_literal ()
// CHECK:STDOUT:     %.loc24_31.2: type = converted %.loc24_31.1, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:     %return: ref %.20 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%U.loc6: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 0 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%U.1) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%U.1) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @I.%.1 (%.5) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self (%Self) = binding_pattern self
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_14.1: @F.1.%I.type (%I.type.2) = specific_constant @I.%Self.1, @I(constants.%U) [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%I.type (%I.type.2) = name_ref Self, %.loc7_14.1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.3: type = converted %Self.ref, %.loc7_14.2 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %self.param: @F.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:       %self: @F.1.%Self (%Self) = bind_name self, %self.param
// CHECK:STDOUT:       %U.ref: type = name_ref U, @I.%U.loc6 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %return: ref @F.1.%U (%U) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7: @I.%.1 (%.5) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc7
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(%V.loc10: type) {
// CHECK:STDOUT:   %V.1: type = bind_symbolic_name V, 0 [symbolic = %V.1 (constants.%V)]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%V.1) [symbolic = %A.1 (constants.%A.3)]
// CHECK:STDOUT:   %I.type.1: type = interface_type @I, @I(%V.1) [symbolic = %I.type.1 (constants.%I.type.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.loc10 as %I.type.loc10 {
// CHECK:STDOUT:     %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:       %self.patt: @F.2.%A (%A.3) = binding_pattern self
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, @impl.%A.loc10 [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:       %self.param: @F.2.%A (%A.3) = param self, runtime_param0
// CHECK:STDOUT:       %self: @F.2.%A (%A.3) = bind_name self, %self.param
// CHECK:STDOUT:       %V.ref: type = name_ref V, @impl.%V.loc10 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:       %return: ref @F.2.%V (%V) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: <witness> = interface_witness (%F.decl) [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc2: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.1) [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @A.%A (%A.2), @A.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = struct_type {.n: @A.%T.1 (%T)} [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness @A.%.2 (%.3) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc2 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc3: @A.%.1 (%.2) = field_decl n, element0 [template]
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.3 [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2
// CHECK:STDOUT:     .n = %.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%U.loc6: type, @I.%Self.1: @I.%I.type (%I.type.2)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(%U) [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.1.%Self (%Self)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@impl.%V.loc10: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V, 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @F.2.%A (%A.3), @F.2.%V (%V) [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.2.%A (%A.3)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.3) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @F.2.%.1 (%.10) = name_ref n, @A.%.loc3 [template = @A.%.loc3]
// CHECK:STDOUT:     %.loc12_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc12_16.2: @F.2.%V (%V) = bind_value %.loc12_16.1
// CHECK:STDOUT:     return %.loc12_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc16: type) {
// CHECK:STDOUT:   %W.1: type = bind_symbolic_name W, 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%W.1) [symbolic = %A.1 (constants.%A.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.1: type = interface_type @I, @I(%W.1) [symbolic = %I.type.1 (constants.%I.type.4)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%W.1) [symbolic = %F.type (constants.%F.type.4)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @TestGeneric.%I.type.1 (%I.type.4), @TestGeneric.%F.type (%F.type.4) [symbolic = %.1 (constants.%.18)]
// CHECK:STDOUT:   %.2: @TestGeneric.%.1 (%.18) = assoc_entity element0, @I.%F.decl [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.loc16: type](%a: @TestGeneric.%A.1 (%A.4)) -> @TestGeneric.%W.1 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A.1 (%A.4) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %W.ref.loc21: type = name_ref W, %W.loc16 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:     %I.type.loc21: type = interface_type @I, @I(constants.%W) [symbolic = %I.type.1 (constants.%I.type.4)]
// CHECK:STDOUT:     %.loc21: @TestGeneric.%.1 (%.18) = specific_constant @I.%.loc7, @I(constants.%W) [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%.1 (%.18) = name_ref F, %.loc21 [symbolic = %.2 (constants.%.19)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a: %A.5) -> %.20 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.5 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:   %.loc28_16: %.20 = struct_literal ()
// CHECK:STDOUT:   %.loc28_14: type = converted %.loc28_16, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:   %I.type: type = interface_type @I, @I(constants.%.20) [template = constants.%I.type.5]
// CHECK:STDOUT:   %.loc28_18: %.26 = specific_constant @I.%.loc7, @I(constants.%.20) [template = constants.%.27]
// CHECK:STDOUT:   %F.ref: %.26 = name_ref F, %.loc28_18 [template = constants.%.27]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@A.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%Self) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%U.1) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%V) {
// CHECK:STDOUT:   %T.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT:   %.3 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.%V.1) {
// CHECK:STDOUT:   %T.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%V.1) {
// CHECK:STDOUT:   %U.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V.1 => constants.%V
// CHECK:STDOUT:   %A.1 => constants.%A.3
// CHECK:STDOUT:   %I.type.1 => constants.%I.type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.2.%V) {
// CHECK:STDOUT:   %T.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%V, constants.%A.3) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %I.type => constants.%I.type.3
// CHECK:STDOUT:   %Self => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT:   %.3 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@TestGeneric.%W.1) {
// CHECK:STDOUT:   %T.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT:   %A.1 => constants.%A.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U.1 => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%W.1) {
// CHECK:STDOUT:   %U.1 => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%.20) {
// CHECK:STDOUT:   %T.1 => constants.%.20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.5
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT:   %.3 => constants.%.23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.20) {
// CHECK:STDOUT:   %U.1 => constants.%.20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.5
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5
// CHECK:STDOUT:   %F => constants.%F.5
// CHECK:STDOUT:   %.1 => constants.%.26
// CHECK:STDOUT:   %.2 => constants.%.27
// CHECK:STDOUT: }
// CHECK:STDOUT:
