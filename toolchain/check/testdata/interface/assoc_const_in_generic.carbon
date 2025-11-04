// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/assoc_const_in_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/assoc_const_in_generic.carbon

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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.070: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.269: %I.type.070 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.533: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %I.F.type.76d: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.dde: %I.F.type.76d = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.b65: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.b4f: %I.assoc_type.b65 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c94: <witness> = require_complete_type %I.type.070 [symbolic]
// CHECK:STDOUT:   %require_complete.90f: <witness> = require_complete_type %I.assoc_type.b65 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.type.399: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.32d: %I.type.399 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.684: type = fn_type @I.F, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.F.a8d: %I.F.type.684 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.22c: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.a7f: %I.assoc_type.22c = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.e4b: <witness> = complete_type_witness %I.type.399 [concrete]
// CHECK:STDOUT:   %complete_type.d28: <witness> = complete_type_witness %I.assoc_type.22c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc19_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.loc15_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc15_13.1)> [symbolic = %I.type (constants.%I.type.070)]
// CHECK:STDOUT:   %Self.loc15_23.2: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc15_23.2 (constants.%Self.269)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T.loc15_13.1) [symbolic = %I.F.type (constants.%I.F.type.76d)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.76d) = struct_value () [symbolic = %I.F (constants.%I.F.dde)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc15_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0.loc16_22.2: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc16_22.2 (constants.%assoc0.b4f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_23.1: @I.%I.type (%I.type.070) = symbolic_binding Self, 1 [symbolic = %Self.loc15_23.2 (constants.%Self.269)]
// CHECK:STDOUT:     %I.F.decl: @I.%I.F.type (%I.F.type.76d) = fn_decl @I.F [symbolic = @I.%I.F (constants.%I.F.dde)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %return.patt: @I.F.%pattern_type (%pattern_type.533) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @I.F.%pattern_type (%pattern_type.533) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc16_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @I.F.%U.loc16_8.1 (%U) = out_param call_param0
// CHECK:STDOUT:       %return: ref @I.F.%U.loc16_8.1 (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_22.1: @I.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc16_22.2 (constants.%assoc0.b4f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_23.1
// CHECK:STDOUT:     .F = %assoc0.loc16_22.1
// CHECK:STDOUT:     witness = (%I.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%T.loc15_13.2: type, @I.%Self.loc15_23.1: @I.%I.type (%I.type.070), %U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc16_8.1 [symbolic = %pattern_type (constants.%pattern_type.533)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @I.F.%U.loc16_8.1 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19_6.2: type) {
// CHECK:STDOUT:   %T.loc19_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc23_6.2: type = facet_type <@I, @I(%T.loc19_6.1)> [symbolic = %I.type.loc23_6.2 (constants.%I.type.070)]
// CHECK:STDOUT:   %require_complete.loc23_7.1: <witness> = require_complete_type %I.type.loc23_6.2 [symbolic = %require_complete.loc23_7.1 (constants.%require_complete.c94)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc19_6.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.b65)]
// CHECK:STDOUT:   %assoc0: @G.%I.assoc_type (%I.assoc_type.b65) = assoc_entity element0, @I.%I.F.decl [symbolic = %assoc0 (constants.%assoc0.b4f)]
// CHECK:STDOUT:   %require_complete.loc23_7.2: <witness> = require_complete_type %I.assoc_type [symbolic = %require_complete.loc23_7.2 (constants.%require_complete.90f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc23_6.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc23_6.2 (constants.%I.type.070)]
// CHECK:STDOUT:     %.loc23: @G.%I.assoc_type (%I.assoc_type.b65) = specific_constant @I.%assoc0.loc16_22.1, @I(constants.%T) [symbolic = %assoc0 (constants.%assoc0.b4f)]
// CHECK:STDOUT:     %F.ref: @G.%I.assoc_type (%I.assoc_type.b65) = name_ref F, %.loc23 [symbolic = %assoc0 (constants.%assoc0.b4f)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc27_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc27_7: type = converted %.loc27_6, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%empty_struct_type) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.070
// CHECK:STDOUT:   %Self.loc15_23.2 => constants.%Self.269
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.76d
// CHECK:STDOUT:   %I.F => constants.%I.F.dde
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.b65
// CHECK:STDOUT:   %assoc0.loc16_22.2 => constants.%assoc0.b4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.269, constants.%U) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.533
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc23_6.2 => constants.%I.type.399
// CHECK:STDOUT:   %require_complete.loc23_7.1 => constants.%complete_type.e4b
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.22c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a7f
// CHECK:STDOUT:   %require_complete.loc23_7.2 => constants.%complete_type.d28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.399
// CHECK:STDOUT:   %Self.loc15_23.2 => constants.%Self.32d
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.684
// CHECK:STDOUT:   %I.F => constants.%I.F.a8d
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.22c
// CHECK:STDOUT:   %assoc0.loc16_22.2 => constants.%assoc0.a7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
