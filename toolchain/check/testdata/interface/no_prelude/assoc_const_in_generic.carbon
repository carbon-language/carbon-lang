// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_generic.carbon

interface I(T:! type) {
  fn F(U:! type) -> U;
}

fn G(T:! type) {
  // This should not result in a `fn_decl` instruction being added to the eval
  // block for the generic G. This used to crash when printing formatted SemIR
  // because the same instruction ended up in multiple scopes.
  I(T).F;
}

fn H() {
  G({});
}

// CHECK:STDOUT: --- assoc_const_in_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = interface_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %Self: %.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.2, %F.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = interface_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %T.patt.loc11: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc11, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc15: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = param_pattern %T.patt.loc15, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15: type = bind_symbolic_name T, 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I.%.1 (%.2), @I.%F.type (%F.type) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @I.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%.1 (%.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = @I.%F (constants.%F)] {
// CHECK:STDOUT:       %U.patt.loc12: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = param_pattern %U.patt.loc12, runtime_param<invalid> [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, %U.loc12 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:       %return: ref @F.%U.1 (%U) = var <return slot>
// CHECK:STDOUT:       %param: type = param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc12: type = bind_symbolic_name U, 2, %param [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12: @I.%.2 (%.3) = assoc_entity element0, %F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %.loc12
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc11: type, @I.%Self.1: @I.%.1 (%.2), %U.loc12: type) {
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U, 2 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.1: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: type) -> @F.%U.1 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc15: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = interface_type @I, @I(%T.1) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @G.%.1 (%.2), @G.%F.type (%F.type) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:   %.3: @G.%.2 (%.3) = assoc_entity element0, @I.%F.decl [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc19_4: type = interface_type @I, @I(constants.%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:     %.loc19_7: @G.%.2 (%.3) = specific_constant @I.%.loc12, @I(constants.%T) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:     %F.ref: @G.%.2 (%.3) = name_ref F, %.loc19_7 [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc23_6: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc23_4: type = converted %.loc23_6, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %G.call: init %.1 = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT:   %.3 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self, constants.%U) {
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %U.patt.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@I.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@G.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%.5) {
// CHECK:STDOUT:   %T.1 => constants.%.5
// CHECK:STDOUT:   %T.patt.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
