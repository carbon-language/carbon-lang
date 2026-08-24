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
// CHECK:STDOUT:   %.e63: @I.%.loc16_35.2 (@I.%.loc16_35.2) = splice_inst @I.%.loc16_35.1 [template]
// CHECK:STDOUT:   %.c68: type = splice_inst @I.%.loc16_35.4 [template]
// CHECK:STDOUT:   %pattern_type.6d1: type = pattern_type %.c68 [template]
// CHECK:STDOUT:   %N.patt.718: %pattern_type.6d1 = symbolic_binding_pattern N, 1 [template]
// CHECK:STDOUT:   %N: %.c68 = symbolic_binding N, 1 [template]
// CHECK:STDOUT:   %.299: <instruction> = access_member_action %T, I1 [template]
// CHECK:STDOUT:   %.912: type = type_of_inst %.299 [template]
// CHECK:STDOUT:   %.a7e: %.912 = splice_inst %.299 [template]
// CHECK:STDOUT:   %.32e: <instruction> = convert_to_value_action %.a7e, type [template]
// CHECK:STDOUT:   %.408: type = splice_inst %.32e [template]
// CHECK:STDOUT:   %pattern_type.d1e: type = pattern_type %.408 [template]
// CHECK:STDOUT:   %N.patt.47e: %pattern_type.d1e = symbolic_binding_pattern N, 1 [template]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.223: type = facet_type <@I, @I(%T, %N)> [template]
// CHECK:STDOUT:   %Self: %I.type.223 = symbolic_binding Self, 2 [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T, %N) [template]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc16_23.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc16_32.1: @I.%pattern_type (%pattern_type.6d1) = symbolic_binding_pattern N, 1 [template = %N.patt.loc16_32.2 (constants.%N.patt.718)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_25.1: type = splice_block %.loc16_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc16_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc16_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_23.2: type = symbolic_binding T, 0, template [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc16_35.6: type = splice_block %.loc16_35.8 [template = %.loc16_35.5 (constants.%.c68)] {
// CHECK:STDOUT:       %.Self.frozen.loc16_32: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc16_23.2 [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:       %.loc16_35.7: @I.%.loc16_35.2 (@I.%.loc16_35.2) = splice_inst %.loc16_35.1 [template = %.loc16_35.3 (constants.%.e63)]
// CHECK:STDOUT:       %.loc16_35.8: type = splice_inst %.loc16_35.4 [template = %.loc16_35.5 (constants.%.c68)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc16_32.2: @I.%.loc16_35.5 (%.c68) = symbolic_binding N, 1 [template = %N.loc16_32.1 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc16_23.2: type, %N.loc16_32.2: @I.%.loc16_35.5 (%.c68)) {
// CHECK:STDOUT:   %T.patt.loc16_23.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc16_23.1: type = symbolic_binding T, 0, template [template = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:   %.loc16_35.1: <instruction> = access_member_action %T.ref, I1 [template]
// CHECK:STDOUT:   %.loc16_35.2: type = type_of_inst %.loc16_35.1 [template]
// CHECK:STDOUT:   %.loc16_35.3: @I.%.loc16_35.2 (@I.%.loc16_35.2) = splice_inst %.loc16_35.1 [template = %.loc16_35.3 (constants.%.e63)]
// CHECK:STDOUT:   %.loc16_35.4: <instruction> = convert_to_value_action %.loc16_35.7, type [template]
// CHECK:STDOUT:   %.loc16_35.5: type = splice_inst %.loc16_35.4 [template = %.loc16_35.5 (constants.%.c68)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc16_35.5 [template = %pattern_type (constants.%pattern_type.6d1)]
// CHECK:STDOUT:   %N.patt.loc16_32.2: @I.%pattern_type (%pattern_type.6d1) = symbolic_binding_pattern N, 1 [template = %N.patt.loc16_32.2 (constants.%N.patt.718)]
// CHECK:STDOUT:   %N.loc16_32.1: @I.%.loc16_35.5 (%.c68) = symbolic_binding N, 1 [template = %N.loc16_32.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc16_23.1, %N.loc16_32.1)> [template = %I.type (constants.%I.type.223)]
// CHECK:STDOUT:   %Self.loc16_40.2: @I.%I.type (%I.type.223) = symbolic_binding Self, 2 [template = %Self.loc16_40.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc16_40.1: @I.%I.type (%I.type.223) = symbolic_binding Self, 2 [template = %Self.loc16_40.2 (constants.%Self)]
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
// CHECK:STDOUT:   %.loc16_35.1 => constants.%.299
// CHECK:STDOUT:   %.loc16_35.2 => constants.%.912
// CHECK:STDOUT:   %.loc16_35.3 => constants.%.a7e
// CHECK:STDOUT:   %.loc16_35.4 => constants.%.32e
// CHECK:STDOUT:   %.loc16_35.5 => constants.%.408
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d1e
// CHECK:STDOUT:   %N.patt.loc16_32.2 => constants.%N.patt.47e
// CHECK:STDOUT:   %N.loc16_32.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%N, constants.%Self) {}
// CHECK:STDOUT:
