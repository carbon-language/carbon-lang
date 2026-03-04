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
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.1ab: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fdb: %I.type.1ab = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 2 [symbolic]
// CHECK:STDOUT:   %.f4a: Core.Form = init_form %U, call_param0 [symbolic]
// CHECK:STDOUT:   %pattern_type.62e: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.1d6: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.fdb) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.cd4: %I.WithSelf.F.type.1d6 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.203: %I.assoc_type.76c = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e36: <witness> = require_complete_type %I.type.1ab [symbolic]
// CHECK:STDOUT:   %require_complete.ac0: <witness> = require_complete_type %I.assoc_type.76c [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %I.type.ab5: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %Self.7ae: %I.type.ab5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.3f5: type = fn_type @I.WithSelf.F, @I.WithSelf(%empty_struct_type, %Self.fdb) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.02c: %I.WithSelf.F.type.3f5 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.294: type = assoc_entity_type @I, @I(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %assoc0.c64: %I.assoc_type.294 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %complete_type.5b1: <witness> = complete_type_witness %I.type.ab5 [concrete]
// CHECK:STDOUT:   %complete_type.7fc: <witness> = complete_type_witness %I.assoc_type.294 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_17.1: type = splice_block %.loc15_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_10.1: type = splice_block %.loc19_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc19_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.loc15_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc15_13.1)> [symbolic = %I.type (constants.%I.type.1ab)]
// CHECK:STDOUT:   %Self.loc15_23.2: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15_23.2 (constants.%Self.fdb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_23.1: @I.%I.type (%I.type.1ab) = symbolic_binding Self, 1 [symbolic = %Self.loc15_23.2 (constants.%Self.fdb)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.1d6) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.cd4)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 2 [concrete]
// CHECK:STDOUT:       %return.patt: @I.WithSelf.F.%pattern_type (%pattern_type.62e) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @I.WithSelf.F.%pattern_type (%pattern_type.62e) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, %U.loc16_8.2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:       %.loc16_21.2: Core.Form = init_form %U.ref, call_param0 [symbolic = %.loc16_21.1 (constants.%.f4a)]
// CHECK:STDOUT:       %.loc16_12.1: type = splice_block %.loc16_12.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc16_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc16_8.2: type = symbolic_binding U, 2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @I.WithSelf.F.%U.loc16_8.1 (%U) = out_param call_param0
// CHECK:STDOUT:       %return: ref @I.WithSelf.F.%U.loc16_8.1 (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_22.1: @I.WithSelf.%I.assoc_type (%I.assoc_type.76c) = assoc_entity element0, %I.WithSelf.F.decl [symbolic = %assoc0.loc16_22.2 (constants.%assoc0.203)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_23.1
// CHECK:STDOUT:     .F = @I.WithSelf.%assoc0.loc16_22.1
// CHECK:STDOUT:     witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%T.loc15_13.2: type, @I.%Self.loc15_23.1: @I.%I.type (%I.type.1ab), %U.loc16_8.2: type) {
// CHECK:STDOUT:   %U.loc16_8.1: type = symbolic_binding U, 2 [symbolic = %U.loc16_8.1 (constants.%U)]
// CHECK:STDOUT:   %.loc16_21.1: Core.Form = init_form %U.loc16_8.1, call_param0 [symbolic = %.loc16_21.1 (constants.%.f4a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc16_8.1 [symbolic = %pattern_type (constants.%pattern_type.62e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @I.WithSelf.F.%U.loc16_8.1 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc19_6.2: type) {
// CHECK:STDOUT:   %T.loc19_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc23_6.2: type = facet_type <@I, @I(%T.loc19_6.1)> [symbolic = %I.type.loc23_6.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:   %require_complete.loc23_7.1: <witness> = require_complete_type %I.type.loc23_6.2 [symbolic = %require_complete.loc23_7.1 (constants.%require_complete.e36)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc19_6.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.76c)]
// CHECK:STDOUT:   %assoc0: @G.%I.assoc_type (%I.assoc_type.76c) = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.203)]
// CHECK:STDOUT:   %require_complete.loc23_7.2: <witness> = require_complete_type %I.assoc_type [symbolic = %require_complete.loc23_7.2 (constants.%require_complete.ac0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc23_6.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc23_6.2 (constants.%I.type.1ab)]
// CHECK:STDOUT:     %.loc23: @G.%I.assoc_type (%I.assoc_type.76c) = specific_constant @I.WithSelf.%assoc0.loc16_22.1, @I.WithSelf(constants.%T, constants.%Self.fdb) [symbolic = %assoc0 (constants.%assoc0.203)]
// CHECK:STDOUT:     %F.ref: @G.%I.assoc_type (%I.assoc_type.76c) = name_ref F, %.loc23 [symbolic = %assoc0 (constants.%assoc0.203)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc27_6: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self.loc15_23.2 => constants.%Self.fdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.1ab
// CHECK:STDOUT:   %Self => constants.%Self.fdb
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1d6
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.cd4
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.76c
// CHECK:STDOUT:   %assoc0.loc16_22.2 => constants.%assoc0.203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.fdb, constants.%U) {
// CHECK:STDOUT:   %U.loc16_8.1 => constants.%U
// CHECK:STDOUT:   %.loc16_21.1 => constants.%.f4a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.62e
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
// CHECK:STDOUT:   %I.type.loc23_6.2 => constants.%I.type.ab5
// CHECK:STDOUT:   %require_complete.loc23_7.1 => constants.%complete_type.5b1
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.294
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c64
// CHECK:STDOUT:   %require_complete.loc23_7.2 => constants.%complete_type.7fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %Self.loc15_23.2 => constants.%Self.7ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%empty_struct_type, constants.%Self.fdb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%empty_struct_type
// CHECK:STDOUT:   %I.type => constants.%I.type.ab5
// CHECK:STDOUT:   %Self => constants.%Self.fdb
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.3f5
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.02c
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.294
// CHECK:STDOUT:   %assoc0.loc16_22.2 => constants.%assoc0.c64
// CHECK:STDOUT: }
// CHECK:STDOUT:
