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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/basic.carbon

interface Empty {
}

interface ForwardDeclared;

interface ForwardDeclared {
  fn F();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.61e: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.8e7: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared+Self.F.type: type = fn_type @ForwardDeclared+Self.F, @ForwardDeclared(%Self.8e7) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared+Self.F: %ForwardDeclared+Self.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.%ForwardDeclared+Self.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .ForwardDeclared = %ForwardDeclared.decl.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc18: type = interface_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT:   %ForwardDeclared.decl.loc20: type = interface_decl @ForwardDeclared [concrete = constants.%ForwardDeclared.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.61e]
// CHECK:STDOUT:   interface_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self.loc20: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8e7]
// CHECK:STDOUT:   interface_with_self_decl @ForwardDeclared [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %ForwardDeclared+Self.F.decl: @ForwardDeclared.%ForwardDeclared+Self.F.type (%ForwardDeclared+Self.F.type) = fn_decl @ForwardDeclared+Self.F [symbolic = @ForwardDeclared.%ForwardDeclared+Self.F (constants.%ForwardDeclared+Self.F)] {} {}
// CHECK:STDOUT:   %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, %ForwardDeclared+Self.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc20
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%ForwardDeclared+Self.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared+Self.F(@ForwardDeclared.%Self.loc20: %ForwardDeclared.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%Self.61e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared(constants.%Self.8e7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared+Self.F(constants.%Self.8e7) {}
// CHECK:STDOUT:
