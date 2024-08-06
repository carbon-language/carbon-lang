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
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I[%Self] [symbolic]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @I.%U [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @I[%Self] [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %.1, %G.type [symbolic]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element2, @I.%G.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @I[%Self.2] [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.1, @I.%F.type (%F.type) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:   %.2: @I.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @I[%Self.2] [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @I.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, @I.%G.type (%G.type) [symbolic = %.3 (constants.%.7)]
// CHECK:STDOUT:   %.4: @I.%.3 (%.7) = assoc_entity element2, %G.decl [symbolic = %.4 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self)]
// CHECK:STDOUT:   %F.decl: @I.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {
// CHECK:STDOUT:     %T.loc12_8.1: type = param T
// CHECK:STDOUT:     %T.loc12_8.2: type = bind_symbolic_name T 1, %T.loc12_8.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: @I.%.1 (%.3) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.4)]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %.loc13: %.5 = assoc_entity element1, %U [template = constants.%.6]
// CHECK:STDOUT:   %G.decl: @I.%G.type (%G.type) = fn_decl @G [symbolic = %G (constants.%G)] {
// CHECK:STDOUT:     %T.loc16_8.1: type = param T
// CHECK:STDOUT:     %T.loc16_8.2: type = bind_symbolic_name T 1, %T.loc16_8.1 [symbolic = @G.%T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: @I.%.3 (%.7) = assoc_entity element2, %G.decl [symbolic = %.4 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .F = %.loc12
// CHECK:STDOUT:   .U = %.loc13
// CHECK:STDOUT:   .G = %.loc16
// CHECK:STDOUT:   witness = (%F.decl, %U, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self.1: %.1, @I.%T.loc12_8.2: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 1 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@I.%T.loc12_8.2: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@I.%Self.1: %.1, @I.%T.loc16_8.2: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 1 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(@I.%T.loc16_8.2: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[constants.%Self] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self, constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I[@I.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
