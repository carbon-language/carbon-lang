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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1: type = generic_interface_type @I [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %I.generic: %I.type.1 = struct_value () [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 2 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 2 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F, @I(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %I.type.2, %F.type.1 [symbolic]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%empty_struct_type) [template]
// CHECK:STDOUT:   %I.type.3: type = facet_type <@I, @I(%empty_struct_type)> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F, @I(%empty_struct_type) [template]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %I.type.3, %F.type.2 [template]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.1 = interface_decl @I [template = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc15_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_6.1, runtime_param<invalid> [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc11_13.2)> [symbolic = %I.type (constants.%I.type.2)]
// CHECK:STDOUT:   %Self.2: %I.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.1) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @I.%I.type (%I.type.2), @I.%F.type (%F.type.1) [symbolic = %F.assoc_type (constants.%F.assoc_type.1)]
// CHECK:STDOUT:   %assoc0.loc12_22.2: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_22.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.1) = fn_decl @F [symbolic = @I.%F (constants.%F)] {
// CHECK:STDOUT:       %U.patt.loc12_8.2: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc12_8.1 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc12_8.2, runtime_param<invalid> [symbolic = %U.patt.loc12_8.1 (constants.%U.patt)]
// CHECK:STDOUT:       %return.patt: @F.%U.loc12_8.1 (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.%U.loc12_8.1 (%U) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, %U.loc12_8.2 [symbolic = %U.loc12_8.1 (constants.%U)]
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc12_8.2: type = bind_symbolic_name U, 2, %U.param [symbolic = %U.loc12_8.1 (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @F.%U.loc12_8.1 (%U) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @F.%U.loc12_8.1 (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_22.1: @I.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc12_22.2 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .F = %assoc0.loc12_22.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%T.loc11_13.1: type, @I.%Self.1: @I.%I.type (%I.type.2), %U.loc12_8.2: type) {
// CHECK:STDOUT:   %U.loc12_8.1: type = bind_symbolic_name U, 2 [symbolic = %U.loc12_8.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_8.1: type = symbolic_binding_pattern U, 2 [symbolic = %U.patt.loc12_8.1 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: type) -> @F.%U.loc12_8.1 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc15_6.1: type) {
// CHECK:STDOUT:   %T.loc15_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc19_6.2: type = facet_type <@I, @I(%T.loc15_6.2)> [symbolic = %I.type.loc19_6.2 (constants.%I.type.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I(%T.loc15_6.2) [symbolic = %F.type (constants.%F.type.1)]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type @G.%I.type.loc19_6.2 (%I.type.2), @G.%F.type (%F.type.1) [symbolic = %F.assoc_type (constants.%F.assoc_type.1)]
// CHECK:STDOUT:   %assoc0: @G.%F.assoc_type (%F.assoc_type.1) = assoc_entity element0, @I.%F.decl [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %I.ref: %I.type.1 = name_ref I, file.%I.decl [template = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_6.1 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:     %I.type.loc19_6.1: type = facet_type <@I, @I(constants.%T)> [symbolic = %I.type.loc19_6.2 (constants.%I.type.2)]
// CHECK:STDOUT:     %.loc19: @G.%F.assoc_type (%F.assoc_type.1) = specific_constant @I.%assoc0.loc12_22.1, @I(constants.%T) [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:     %F.ref: @G.%F.assoc_type (%F.assoc_type.1) = name_ref F, %.loc19 [symbolic = %assoc0 (constants.%assoc0.1)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc23_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc23_7: type = converted %.loc23_6, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%empty_struct_type) [template = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.2
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT:   %F.type => constants.%F.type.1
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.1
// CHECK:STDOUT:   %assoc0.loc12_22.2 => constants.%assoc0.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%Self, constants.%U) {
// CHECK:STDOUT:   %U.loc12_8.1 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_8.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@G.%T.loc15_6.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc15_6.2 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type.loc19_6.2 => constants.%I.type.3
// CHECK:STDOUT:   %F.type => constants.%F.type.2
// CHECK:STDOUT:   %F.assoc_type => constants.%F.assoc_type.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
