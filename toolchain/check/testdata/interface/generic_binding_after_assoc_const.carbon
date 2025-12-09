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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/generic_binding_after_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/generic_binding_after_assoc_const.carbon

interface I {
  fn F(T:! type);
  let U:! type;
  // This `T` should have index 1, just like the `T` above.
  // The intervening `U:! type` should not cause this to have index 2.
  fn G(T:! type);
}

// CHECK:STDOUT: --- generic_binding_after_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete]
// CHECK:STDOUT:   %I.G.type: type = fn_type @I.G [concrete]
// CHECK:STDOUT:   %I.G: %I.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.%I.G.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc16_8.2: type = symbolic_binding T, 1 [symbolic = %T.loc16_8.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.G.decl: %I.G.type = fn_decl @I.G [concrete = constants.%I.G] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc20_8.2: type = symbolic_binding T, 1 [symbolic = %T.loc20_8.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, %I.G.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%I.F.decl, %U, %I.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type, %T.loc16_8.2: type) {
// CHECK:STDOUT:   %T.loc16_8.1: type = symbolic_binding T, 1 [symbolic = %T.loc16_8.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.G(@I.%Self: %I.type, %T.loc20_8.2: type) {
// CHECK:STDOUT:   %T.loc20_8.1: type = symbolic_binding T, 1 [symbolic = %T.loc20_8.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.loc16_8.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.G(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.loc20_8.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
