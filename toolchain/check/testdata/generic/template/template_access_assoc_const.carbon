// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/template_access_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/template_access_assoc_const.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
interface I(template T: type, N: T.I1) {
    let I1: type;
}
//@dump-sem-ir-end

// TODO: add code to demonstrate that T.I1 is being properly handled
// here. Currently this crashes:
//
// class C {
//   alias I1 = type;
// }
// impl C as I(C, C) where .I1 = C {}

// CHECK:STDOUT: --- template_access_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %.eec: type = splice_inst @I.%.loc16_35.1 [template]
// CHECK:STDOUT:   %.4a7: type = type_of_inst @I.%.loc16_35.3 [template]
// CHECK:STDOUT:   %.e35: %.4a7 = splice_inst @I.%.loc16_35.3 [template]
// CHECK:STDOUT:   %.476: type = splice_inst @I.%.loc16_35.6 [template]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %.476 [template]
// CHECK:STDOUT:   %N.patt.9e2: %pattern_type.8f9 = symbolic_binding_pattern N, 1 [template]
// CHECK:STDOUT:   %N: %.476 = symbolic_binding N, 1 [template]
// CHECK:STDOUT:   %.dc4: <instruction> = refine_inst_action %T [template]
// CHECK:STDOUT:   %.40e: type = splice_inst %.dc4 [template]
// CHECK:STDOUT:   %.2e4: <instruction> = access_member_action %.40e, I1 [template]
// CHECK:STDOUT:   %.ffd: type = type_of_inst %.2e4 [template]
// CHECK:STDOUT:   %.242: %.ffd = splice_inst %.2e4 [template]
// CHECK:STDOUT:   %.626: <instruction> = convert_to_value_action %.242, type [template]
// CHECK:STDOUT:   %.e0e: type = splice_inst %.626 [template]
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %.e0e [template]
// CHECK:STDOUT:   %N.patt.f4e: %pattern_type.55f = symbolic_binding_pattern N, 1 [template]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.f82: type = facet_type <@I, @I(%T, %N)> [template]
// CHECK:STDOUT:   %Self: %I.type.f82 = symbolic_binding Self, 2 [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T, %N) [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc16_23.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc16_32.1: @I.%pattern_type (%pattern_type.8f9) = symbolic_binding_pattern N, 1 [template = %N.patt.loc16_32.2 (constants.%N.patt.9e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_25.1: type = splice_block %.loc16_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc16_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc16_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_23.2: type = symbolic_binding T, 0, template [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc16_35.8: type = splice_block %.loc16_35.12 [template = %.loc16_35.7 (constants.%.476)] {
// CHECK:STDOUT:       %.Self.frozen.loc16_32: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc16_23.2 [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:       %.loc16_35.9: type = splice_inst %.loc16_35.1 [template = %.loc16_35.2 (constants.%.eec)]
// CHECK:STDOUT:       %.loc16_35.10: type = type_of_inst %.loc16_35.3 [template = %.loc16_35.4 (constants.%.4a7)]
// CHECK:STDOUT:       %.loc16_35.11: @I.%.loc16_35.4 (%.4a7) = splice_inst %.loc16_35.3 [template = %.loc16_35.5 (constants.%.e35)]
// CHECK:STDOUT:       %.loc16_35.12: type = splice_inst %.loc16_35.6 [template = %.loc16_35.7 (constants.%.476)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc16_32.2: @I.%.loc16_35.7 (%.476) = symbolic_binding N, 1 [template = %N.loc16_32.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc16_23.2: type, %N.loc16_32.2: @I.%.loc16_35.7 (%.476)) {
// CHECK:STDOUT:   %T.patt.loc16_23.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc16_23.1: type = symbolic_binding T, 0, template [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:   %.loc16_35.1: <instruction> = refine_inst_action %T.ref [template]
// CHECK:STDOUT:   %.loc16_35.2: type = splice_inst %.loc16_35.1 [template = %.loc16_35.2 (constants.%.eec)]
// CHECK:STDOUT:   %.loc16_35.3: <instruction> = access_member_action %.loc16_35.9, I1 [template]
// CHECK:STDOUT:   %.loc16_35.4: type = type_of_inst %.loc16_35.3 [template = %.loc16_35.4 (constants.%.4a7)]
// CHECK:STDOUT:   %.loc16_35.5: @I.%.loc16_35.4 (%.4a7) = splice_inst %.loc16_35.3 [template = %.loc16_35.5 (constants.%.e35)]
// CHECK:STDOUT:   %.loc16_35.6: <instruction> = convert_to_value_action %.loc16_35.11, type [template]
// CHECK:STDOUT:   %.loc16_35.7: type = splice_inst %.loc16_35.6 [template = %.loc16_35.7 (constants.%.476)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc16_35.7 [template = %pattern_type (constants.%pattern_type.8f9)]
// CHECK:STDOUT:   %N.patt.loc16_32.2: @I.%pattern_type (%pattern_type.8f9) = symbolic_binding_pattern N, 1 [template = %N.patt.loc16_32.2 (constants.%N.patt.9e2)]
// CHECK:STDOUT:   %N.loc16_32.1: @I.%.loc16_35.7 (%.476) = symbolic_binding N, 1 [template = %N.loc16_32.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc16_23.1, %N.loc16_32.1)> [template = %I.type (constants.%I.type.f82)]
// CHECK:STDOUT:   %Self.loc16_40.2: @I.%I.type (%I.type.f82) = symbolic_binding Self, 2 [template = %Self.loc16_40.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc16_40.1: @I.%I.type (%I.type.f82) = symbolic_binding Self, 2 [template = %Self.loc16_40.2 (constants.%Self)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I1: type = assoc_const_decl @I1 [concrete] {
// CHECK:STDOUT:       %assoc0: @I.WithSelf.%I.assoc_type (%I.assoc_type) = assoc_entity element0, @I.WithSelf.%I1 [template = @I.WithSelf.%assoc0 (constants.%assoc0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc16_40.1
// CHECK:STDOUT:     .I1 = @I1.%assoc0
// CHECK:STDOUT:     witness = (@I.WithSelf.%I1)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.patt.loc16_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc16_23.1 => constants.%T
// CHECK:STDOUT:   %.loc16_35.1 => constants.%.dc4
// CHECK:STDOUT:   %.loc16_35.2 => constants.%.40e
// CHECK:STDOUT:   %.loc16_35.3 => constants.%.2e4
// CHECK:STDOUT:   %.loc16_35.4 => constants.%.ffd
// CHECK:STDOUT:   %.loc16_35.5 => constants.%.242
// CHECK:STDOUT:   %.loc16_35.6 => constants.%.626
// CHECK:STDOUT:   %.loc16_35.7 => constants.%.e0e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.55f
// CHECK:STDOUT:   %N.patt.loc16_32.2 => constants.%N.patt.f4e
// CHECK:STDOUT:   %N.loc16_32.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%N, constants.%Self) {}
// CHECK:STDOUT:
