// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/generic_binding_after_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/generic_binding_after_assoc_const.carbon

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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.%G.decl [concrete]
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc12_8.1: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc12_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc12_8.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc12_8.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.%U [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc16_8.1: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc16_8.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc16_8.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc16_8.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, %G.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .U = @U.%assoc1
// CHECK:STDOUT:   .G = %assoc2
// CHECK:STDOUT:   witness = (%F.decl, %U, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type, %T.loc12_8.2: type) {
// CHECK:STDOUT:   %T.loc12_8.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc12_8.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_8.2: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc12_8.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc12_8.1: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@I.%Self: %I.type, %T.loc16_8.2: type) {
// CHECK:STDOUT:   %T.loc16_8.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc16_8.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc16_8.2: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc16_8.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc16_8.1: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.loc12_8.1 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_8.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.loc16_8.1 => constants.%T
// CHECK:STDOUT:   %T.patt.loc16_8.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
