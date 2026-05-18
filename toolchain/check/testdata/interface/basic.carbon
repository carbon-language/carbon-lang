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
// CHECK:STDOUT:   %Self.63a: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.type: type = facet_type <@ForwardDeclared> [concrete]
// CHECK:STDOUT:   %Self.b75: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.type: type = fn_type @ForwardDeclared.WithSelf.F, @ForwardDeclared.WithSelf(%Self.b75) [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F: %ForwardDeclared.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ForwardDeclared.assoc_type: type = assoc_entity_type @ForwardDeclared [concrete]
// CHECK:STDOUT:   %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.decl [concrete]
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
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.63a]
// CHECK:STDOUT:   %Empty.WithSelf.decl = interface_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @ForwardDeclared {
// CHECK:STDOUT:   %Self: %ForwardDeclared.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b75]
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.decl = interface_with_self_decl @ForwardDeclared [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %ForwardDeclared.WithSelf.F.decl: @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.type (%ForwardDeclared.WithSelf.F.type) = fn_decl @ForwardDeclared.WithSelf.F [symbolic = @ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F (constants.%ForwardDeclared.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %ForwardDeclared.assoc_type = assoc_entity element0, %ForwardDeclared.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @ForwardDeclared.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@ForwardDeclared.WithSelf.%ForwardDeclared.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ForwardDeclared.WithSelf.F(@ForwardDeclared.%Self: %ForwardDeclared.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.63a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf(constants.%Self.b75) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclared.WithSelf.F(constants.%Self.b75) {}
// CHECK:STDOUT:
