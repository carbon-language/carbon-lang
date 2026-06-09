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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/self.carbon

interface UseSelf {
  fn F(self) -> Self;
}


// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UseSelf.type: type = facet_type <@UseSelf> [concrete]
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %.326: Core.Form = init_form %Self.as_type [symbolic]
// CHECK:STDOUT:   %UseSelf.WithSelf.F.type: type = fn_type @UseSelf.WithSelf.F, @UseSelf.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %UseSelf.WithSelf.F: %UseSelf.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %UseSelf.assoc_type: type = assoc_entity_type @UseSelf [concrete]
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, @UseSelf.WithSelf.%UseSelf.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [concrete = constants.%UseSelf.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %UseSelf.WithSelf.decl = interface_with_self_decl @UseSelf [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %UseSelf.WithSelf.F.decl: @UseSelf.WithSelf.%UseSelf.WithSelf.F.type (%UseSelf.WithSelf.F.type) = fn_decl @UseSelf.WithSelf.F [symbolic = @UseSelf.WithSelf.%UseSelf.WithSelf.F (constants.%UseSelf.WithSelf.F)] {
// CHECK:STDOUT:     %self.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = return_slot_pattern %return.param_patt, %.loc16_17.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_17: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc16_17: type = facet_access_type %Self.ref.loc16_17 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc16_17.2: type = converted %Self.ref.loc16_17, %Self.as_type.loc16_17 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc16_17.3: Core.Form = init_form %.loc16_17.2 [symbolic = %.loc16_17.1 (constants.%.326)]
// CHECK:STDOUT:     %self.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16_8.1: type = splice_block %.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc16_8: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc16_8.2: type = facet_access_type %Self.ref.loc16_8 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc16_8.2: type = converted %Self.ref.loc16_8, %Self.as_type.loc16_8.2 [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %UseSelf.assoc_type = assoc_entity element0, %UseSelf.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @UseSelf.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@UseSelf.WithSelf.%UseSelf.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseSelf.WithSelf.F(@UseSelf.%Self: %UseSelf.type) {
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc16_8.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_8.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc16_8.1 [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %.loc16_17.1: Core.Form = init_form %Self.as_type.loc16_8.1 [symbolic = %.loc16_17.1 (constants.%.326)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type)) -> out %return.param: @UseSelf.WithSelf.F.%Self.as_type.loc16_8.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc16_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %.loc16_17.1 => constants.%.326
// CHECK:STDOUT: }
// CHECK:STDOUT:
