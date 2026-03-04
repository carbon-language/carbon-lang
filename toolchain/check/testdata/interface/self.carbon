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
  fn F[self: Self]() -> Self;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UseSelf.type: type = facet_type <@UseSelf> [concrete]
// CHECK:STDOUT:   %Self: %UseSelf.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %.52b: Core.Form = init_form %Self.binding.as_type, call_param1 [symbolic]
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
// CHECK:STDOUT:     %self.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = value_param_pattern %self.patt [concrete]
// CHECK:STDOUT:     %return.patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @UseSelf.WithSelf.F.%pattern_type (%pattern_type) = out_param_pattern %return.patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_25: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc16_25: type = facet_access_type %Self.ref.loc16_25 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc16_25.2: type = converted %Self.ref.loc16_25, %Self.as_type.loc16_25 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     %.loc16_25.3: Core.Form = init_form %.loc16_25.2, call_param1 [symbolic = %.loc16_25.1 (constants.%.52b)]
// CHECK:STDOUT:     %self.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16_14.1: type = splice_block %.loc16_14.2 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc16_14: %UseSelf.type = name_ref Self, @UseSelf.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc16_14: type = facet_access_type %Self.ref.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       %.loc16_14.2: type = converted %Self.ref.loc16_14, %Self.as_type.loc16_14 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type) = return_slot %return.param
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
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 0, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type)]
// CHECK:STDOUT:   %.loc16_25.1: Core.Form = init_form %Self.binding.as_type, call_param1 [symbolic = %.loc16_25.1 (constants.%.52b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type)) -> out %return.param: @UseSelf.WithSelf.F.%Self.binding.as_type (%Self.binding.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseSelf.WithSelf.F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT:   %.loc16_25.1 => constants.%.52b
// CHECK:STDOUT: }
// CHECK:STDOUT:
