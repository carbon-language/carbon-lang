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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.1: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = interface_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %Self: %.5 = bind_symbolic_name Self 1 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %.5, %F.type.1 [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %V: type = bind_symbolic_name V 0 [symbolic]
// CHECK:STDOUT:   %A.3: type = class_type @A, @A(%V) [symbolic]
// CHECK:STDOUT:   %.8: type = interface_type @I, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.1, @I(%V) [symbolic]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %.8, %F.type.3 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.11: <witness> = interface_witness (%F.2) [template]
// CHECK:STDOUT:   %.12: type = unbound_element_type %A.3, %V [symbolic]
// CHECK:STDOUT:   %.13: type = struct_type {.n: %V} [symbolic]
// CHECK:STDOUT:   %.14: <witness> = complete_type_witness %.13 [symbolic]
// CHECK:STDOUT:   %.15: type = ptr_type %.13 [symbolic]
// CHECK:STDOUT:   %W: type = bind_symbolic_name W 0 [symbolic]
// CHECK:STDOUT:   %A.4: type = class_type @A, @A(%W) [symbolic]
// CHECK:STDOUT:   %TestGeneric.type: type = fn_type @TestGeneric [template]
// CHECK:STDOUT:   %TestGeneric: %TestGeneric.type = struct_value () [template]
// CHECK:STDOUT:   %.16: type = unbound_element_type %A.4, %W [symbolic]
// CHECK:STDOUT:   %.17: type = struct_type {.n: %W} [symbolic]
// CHECK:STDOUT:   %.18: <witness> = complete_type_witness %.17 [symbolic]
// CHECK:STDOUT:   %.19: type = ptr_type %.17 [symbolic]
// CHECK:STDOUT:   %.20: type = interface_type @I, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.1, @I(%W) [symbolic]
// CHECK:STDOUT:   %F.4: %F.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %.20, %F.type.4 [symbolic]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.23: type = struct_type {} [template]
// CHECK:STDOUT:   %A.5: type = class_type @A, @A(%.23) [template]
// CHECK:STDOUT:   %TestSpecific.type: type = fn_type @TestSpecific [template]
// CHECK:STDOUT:   %TestSpecific: %TestSpecific.type = struct_value () [template]
// CHECK:STDOUT:   %.24: type = unbound_element_type %A.5, %.23 [template]
// CHECK:STDOUT:   %.25: type = struct_type {.n: %.23} [template]
// CHECK:STDOUT:   %.26: <witness> = complete_type_witness %.25 [template]
// CHECK:STDOUT:   %.27: type = struct_type {.n: %.1} [template]
// CHECK:STDOUT:   %.28: type = ptr_type %.25 [template]
// CHECK:STDOUT:   %.29: type = interface_type @I, @I(%.23) [template]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.1, @I(%.23) [template]
// CHECK:STDOUT:   %F.5: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.30: type = assoc_entity_type %.29, %F.type.5 [template]
// CHECK:STDOUT:   %.31: %.30 = assoc_entity element0, @I.%F.decl [template]
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
// CHECK:STDOUT:     %T.loc2_9.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc2_9.2: type = bind_symbolic_name T 0, %T.loc2_9.1 [symbolic = @A.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %U.loc6_13.1: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6_13.2: type = bind_symbolic_name U 0, %U.loc6_13.1 [symbolic = @I.%U (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %V.loc10_14.1: type = param V, runtime_param<invalid>
// CHECK:STDOUT:     %V.loc10_14.2: type = bind_symbolic_name V 0, %V.loc10_14.1 [symbolic = @impl.%V (constants.%V)]
// CHECK:STDOUT:     %A.ref.loc10: %A.type = name_ref A, %A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %V.ref.loc10_26: type = name_ref V, %V.loc10_14.2 [symbolic = @impl.%V (constants.%V)]
// CHECK:STDOUT:     %A.loc10: type = class_type @A, @A(constants.%V) [symbolic = @impl.%A (constants.%A.3)]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, %I.decl [template = constants.%I]
// CHECK:STDOUT:     %V.ref.loc10_34: type = name_ref V, %V.loc10_14.2 [symbolic = @impl.%V (constants.%V)]
// CHECK:STDOUT:     %.loc10: type = interface_type @I, @I(constants.%V) [symbolic = @impl.%.1 (constants.%.8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestGeneric.decl: %TestGeneric.type = fn_decl @TestGeneric [template = constants.%TestGeneric] {
// CHECK:STDOUT:     %W.loc16_16.1: type = param W, runtime_param<invalid>
// CHECK:STDOUT:     @TestGeneric.%W.loc16: type = bind_symbolic_name W 0, %W.loc16_16.1 [symbolic = @TestGeneric.%W.1 (constants.%W)]
// CHECK:STDOUT:     %A.ref.loc16: %A.type = name_ref A, %A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %W.ref.loc16_31: type = name_ref W, @TestGeneric.%W.loc16 [symbolic = @TestGeneric.%W.1 (constants.%W)]
// CHECK:STDOUT:     %A.loc16: type = class_type @A, @A(constants.%W) [symbolic = @TestGeneric.%A (constants.%A.4)]
// CHECK:STDOUT:     %a.loc16_26.1: @TestGeneric.%A (%A.4) = param a, runtime_param0
// CHECK:STDOUT:     @TestGeneric.%a: @TestGeneric.%A (%A.4) = bind_name a, %a.loc16_26.1
// CHECK:STDOUT:     %W.ref.loc16_38: type = name_ref W, @TestGeneric.%W.loc16 [symbolic = @TestGeneric.%W.1 (constants.%W)]
// CHECK:STDOUT:     @TestGeneric.%return: ref @TestGeneric.%W.1 (%W) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestSpecific.decl: %TestSpecific.type = fn_decl @TestSpecific [template = constants.%TestSpecific] {
// CHECK:STDOUT:     %A.ref.loc24: %A.type = name_ref A, %A.decl [template = constants.%A.1]
// CHECK:STDOUT:     %.loc24_23: %.23 = struct_literal ()
// CHECK:STDOUT:     %.loc24_21: type = converted %.loc24_23, constants.%.23 [template = constants.%.23]
// CHECK:STDOUT:     %A.loc24: type = class_type @A, @A(constants.%.23) [template = constants.%A.5]
// CHECK:STDOUT:     %a.loc24_17.1: %A.5 = param a, runtime_param0
// CHECK:STDOUT:     @TestSpecific.%a: %A.5 = bind_name a, %a.loc24_17.1
// CHECK:STDOUT:     %.loc24_31.1: %.23 = struct_literal ()
// CHECK:STDOUT:     %.loc24_31.2: type = converted %.loc24_31.1, constants.%.23 [template = constants.%.23]
// CHECK:STDOUT:     @TestSpecific.%return: ref %.23 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(file.%U.loc6_13.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%U) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %Self.2: %.5 = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%U) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F.1)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I.%.1 (%.5), @I.%F.type (%F.type.1) [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:   %.3: @I.%.2 (%.6) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F.1 [symbolic = %F (constants.%F.1)] {
// CHECK:STDOUT:       %.loc7_14.1: @F.1.%.1 (%.5) = specific_constant %Self.1, @I(constants.%U) [symbolic = @F.1.%Self (constants.%Self)]
// CHECK:STDOUT:       %Self.ref: @F.1.%.1 (%.5) = name_ref Self, %.loc7_14.1 [symbolic = @F.1.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.2: type = facet_type_access %Self.ref [symbolic = @F.1.%Self (constants.%Self)]
// CHECK:STDOUT:       %.loc7_14.3: type = converted %Self.ref, %.loc7_14.2 [symbolic = @F.1.%Self (constants.%Self)]
// CHECK:STDOUT:       %self.loc7_8.1: @F.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc7_8.2: @F.1.%Self (%Self) = bind_name self, %self.loc7_8.1
// CHECK:STDOUT:       %U.ref: type = name_ref U, file.%U.loc6_13.2 [symbolic = @F.1.%U (constants.%U)]
// CHECK:STDOUT:       %return.var: ref @F.1.%U (%U) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_26: @I.%.2 (%.6) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc7_26
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(file.%V.loc10_14.2: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%V) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %A.3 as %.8 {
// CHECK:STDOUT:     %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%A.3 [symbolic = @F.2.%A (constants.%A.3)]
// CHECK:STDOUT:       %self.loc11_8.1: @F.2.%A (%A.3) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc11_8.2: @F.2.%A (%A.3) = bind_name self, %self.loc11_8.1
// CHECK:STDOUT:       %V.ref: type = name_ref V, file.%V.loc10_14.2 [symbolic = @F.2.%V (constants.%V)]
// CHECK:STDOUT:       %return.var: ref @F.2.%V (%V) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: <witness> = interface_witness (%F.decl) [template = constants.%.11]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = %.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(file.%T.loc2_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic = %A (constants.%A.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @A.%A (%A.2), @A.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = struct_type {.n: @A.%T (%T)} [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness @A.%.2 (%.3) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc2_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc3: @A.%.1 (%.2) = field_decl n, element0 [template]
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.3 [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2
// CHECK:STDOUT:     .n = %.loc3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(file.%U.loc6_13.2: type, @I.%Self.1: @I.%.1 (%.5)) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%U) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %Self: %.5 = bind_symbolic_name Self 1 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@I.%self.loc7_8.2: @F.1.%Self (%Self)]() -> @F.1.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(file.%V.loc10_14.2: type) {
// CHECK:STDOUT:   %V: type = bind_symbolic_name V 0 [symbolic = %V (constants.%V)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%V) [symbolic = %A (constants.%A.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @F.2.%A (%A.3), @F.2.%V (%V) [symbolic = %.1 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@impl.%self.loc11_8.2: @F.2.%A (%A.3)]() -> @F.2.%V (%V) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.2.%A (%A.3) = name_ref self, @impl.%self.loc11_8.2
// CHECK:STDOUT:     %n.ref: @F.2.%.1 (%.12) = name_ref n, @A.%.loc3 [template = @A.%.loc3]
// CHECK:STDOUT:     %.loc12_16.1: ref @F.2.%V (%V) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc12_16.2: @F.2.%V (%V) = bind_value %.loc12_16.1
// CHECK:STDOUT:     return %.loc12_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestGeneric(%W.loc16: type) {
// CHECK:STDOUT:   %W.1: type = bind_symbolic_name W 0 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%W.1) [symbolic = %A (constants.%A.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%W.1) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1, @I(%W.1) [symbolic = %F.type.1 (constants.%F.type.4)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @TestGeneric.%.1 (%.20), @TestGeneric.%F.type.1 (%F.type.4) [symbolic = %.2 (constants.%.21)]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic = %U.2 (constants.%U)]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.1, @I(%U.2) [symbolic = %F.type.2 (constants.%F.type.1)]
// CHECK:STDOUT:   %.3: type = interface_type @I, @I(%U.2) [symbolic = %.3 (constants.%.5)]
// CHECK:STDOUT:   %Self.5: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.5 (constants.%Self)]
// CHECK:STDOUT:   %F.decl: @TestGeneric.%F.type.2 (%F.type.1) = fn_decl @F.1 [symbolic = %F.decl (constants.%F.1)] {
// CHECK:STDOUT:     %Self.5: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.5 (constants.%Self)]
// CHECK:STDOUT:     %Self.5: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.5 (constants.%Self)]
// CHECK:STDOUT:     %Self.5: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.5 (constants.%Self)]
// CHECK:STDOUT:     %Self.5: @I.%.1 (%.5) = bind_symbolic_name Self 1 [symbolic = %Self.5 (constants.%Self)]
// CHECK:STDOUT:     @I.%self.loc7_8.1: @F.1.%Self (%Self) = param self, runtime_param0
// CHECK:STDOUT:     @I.%self.loc7_8.2: @F.1.%Self (%Self) = bind_name self, @I.%self.loc7_8.1
// CHECK:STDOUT:     %U.2: type = bind_symbolic_name U 0 [symbolic = %U.2 (constants.%U)]
// CHECK:STDOUT:     @I.%return.var: ref @F.1.%U (%U) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.4: @TestGeneric.%.2 (%.21) = assoc_entity element0, %F.decl [symbolic = %.4 (constants.%.22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%W.loc16: type](%a: @TestGeneric.%A (%A.4)) -> @TestGeneric.%W.1 (%W) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @TestGeneric.%A (%A.4) = name_ref a, %a
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %W.ref: type = name_ref W, %W.loc16 [symbolic = %W.1 (constants.%W)]
// CHECK:STDOUT:     %.loc21_14: type = interface_type @I, @I(constants.%W) [symbolic = %.1 (constants.%.20)]
// CHECK:STDOUT:     %.loc21_17: @TestGeneric.%.2 (%.21) = specific_constant @I.%.loc7_26, @I(constants.%W) [symbolic = %.4 (constants.%.22)]
// CHECK:STDOUT:     %F.ref: @TestGeneric.%.2 (%.21) = name_ref F, %.loc21_17 [symbolic = %.4 (constants.%.22)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSpecific(%a: %A.5) -> %.23 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A.5 = name_ref a, %a
// CHECK:STDOUT:   %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:   %.loc28_16: %.23 = struct_literal ()
// CHECK:STDOUT:   %.loc28_14.1: type = converted %.loc28_16, constants.%.23 [template = constants.%.23]
// CHECK:STDOUT:   %.loc28_14.2: type = interface_type @I, @I(constants.%.23) [template = constants.%.29]
// CHECK:STDOUT:   %.loc28_18: %.30 = specific_constant @I.%.loc7_26, @I(constants.%.23) [template = constants.%.31]
// CHECK:STDOUT:   %F.ref: %.30 = name_ref F, %.loc28_18 [template = constants.%.31]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@A.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%Self) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:   %.1 => constants.%.12
// CHECK:STDOUT:   %.2 => constants.%.13
// CHECK:STDOUT:   %.3 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%V) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.3
// CHECK:STDOUT:   %F => constants.%F.3
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT:   %.3 => constants.%.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@impl.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%V) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@F.2.%V) {
// CHECK:STDOUT:   %T => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%V) {
// CHECK:STDOUT:   %V => constants.%V
// CHECK:STDOUT:   %A => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%V, constants.%A.3) {
// CHECK:STDOUT:   %U => constants.%V
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %Self => constants.%A.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%W) {
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT:   %.3 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(@TestGeneric.%W.1) {
// CHECK:STDOUT:   %T => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestGeneric(constants.%W) {
// CHECK:STDOUT:   %W.1 => constants.%W
// CHECK:STDOUT:   %A => constants.%A.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%W) {
// CHECK:STDOUT:   %U => constants.%W
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.20
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.4
// CHECK:STDOUT:   %F => constants.%F.4
// CHECK:STDOUT:   %.2 => constants.%.21
// CHECK:STDOUT:   %.3 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%W.1) {
// CHECK:STDOUT:   %U => constants.%W
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@TestGeneric.%U.2) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%.23) {
// CHECK:STDOUT:   %T => constants.%.23
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.5
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT:   %.2 => constants.%.25
// CHECK:STDOUT:   %.3 => constants.%.26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%.23) {
// CHECK:STDOUT:   %U => constants.%.23
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.29
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.5
// CHECK:STDOUT:   %F => constants.%F.5
// CHECK:STDOUT:   %.2 => constants.%.30
// CHECK:STDOUT:   %.3 => constants.%.31
// CHECK:STDOUT: }
// CHECK:STDOUT:
