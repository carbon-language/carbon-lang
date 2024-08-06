// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/basic.carbon

interface Empty {
}

interface ForwardDeclared;

interface ForwardDeclared {
  fn F();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Empty [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @ForwardDeclared [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @ForwardDeclared[%Self.2] [symbolic]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %.2, %F.type [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, @ForwardDeclared.%F.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [template = constants.%.1] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc14: type = interface_decl @ForwardDeclared [template = constants.%.2] {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc16: type = interface_decl @ForwardDeclared [template = constants.%.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty[%Self.1: %.1] {
// CHECK:STDOUT:   %Self.2: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared[%Self.1: %.2] {
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @ForwardDeclared[%Self.2] [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @ForwardDeclared.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %.2, @ForwardDeclared.%F.type (%F.type) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @ForwardDeclared.%.1 (%.4) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !body:
// CHECK:STDOUT:   %Self.1: %.2 = bind_symbolic_name Self 0 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %F.decl: @ForwardDeclared.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {}
// CHECK:STDOUT:   %.loc17: @ForwardDeclared.%.1 (%.4) = assoc_entity element0, %F.decl [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.1
// CHECK:STDOUT:   .F = %.loc17
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@ForwardDeclared.%Self.1: %.2) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty[constants.%Self.1] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[constants.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared[@ForwardDeclared.%Self.2] {
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
