// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/unimplemented.carbon

// --- fail_todo_unimplemented_operator.carbon

library "[[@TEST_NAME]]";

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent expression.
fn F[template T: type](x: T) {
  // CHECK:STDERR: fail_todo_unimplemented_operator.carbon:[[@LINE+4]]:3: error: cannot dereference operand of non-pointer type `<dependent type>` [DerefOfNonPointer]
  // CHECK:STDERR:   *x.n;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *x.n;
}

// --- fail_todo_unimplemented_value.carbon

library "[[@TEST_NAME]]";

class C {
  var n: i32;
}

// Check that we get a reasonable diagnostic for an unimplemented operation on a
// template dependent value where the type is concrete but determined through
// the template dependent value.
fn F(template c: C) {
  // CHECK:STDERR: fail_todo_unimplemented_value.carbon:[[@LINE+4]]:3: error: cannot dereference operand of non-pointer type `<dependent type>` [DerefOfNonPointer]
  // CHECK:STDERR:   *c.n;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *c.n;
}

// CHECK:STDOUT: --- fail_todo_unimplemented_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [template]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %.b3e: %T = splice_inst @F.%.loc11_5.4 [template]
// CHECK:STDOUT:   %.432: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst @F.%.loc11_5.6 [template]
// CHECK:STDOUT:   %require_complete.681: <witness> = require_complete_type @F.%.loc11_5.7 [template]
// CHECK:STDOUT:   %.263: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst @F.%.loc11_5.9 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc6_25.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc6_25.1 [template = %x.patt.loc6_25.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc6_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc6_16.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc6_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc6_25.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc6_25.2 [template = %x.patt.loc6_25.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %T.loc6_16.1 [template = %require_complete.loc6 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc11_5.4: <instruction> = refine_type_action %x.ref, %T.loc6_16.1 [template]
// CHECK:STDOUT:   %.loc11_5.5: @F.%T.loc6_16.1 (%T) = splice_inst %.loc11_5.4 [template = %.loc11_5.5 (constants.%.b3e)]
// CHECK:STDOUT:   %.loc11_5.6: <instruction> = access_member_action %.loc11_5.1, n [template]
// CHECK:STDOUT:   %.loc11_5.7: type = type_of_inst %.loc11_5.6 [template]
// CHECK:STDOUT:   %.loc11_5.8: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst %.loc11_5.6 [template = %.loc11_5.8 (constants.%.432)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %.loc11_5.7 [template = %require_complete.loc11 (constants.%require_complete.681)]
// CHECK:STDOUT:   %.loc11_5.9: <instruction> = convert_to_category_action %.loc11_5.2, element1 [template]
// CHECK:STDOUT:   %.loc11_5.10: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst %.loc11_5.9 [template = %.loc11_5.10 (constants.%.263)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc6_16.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc6_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc11_5.1: @F.%T.loc6_16.1 (%T) = splice_inst %.loc11_5.4 [template = %.loc11_5.5 (constants.%.b3e)]
// CHECK:STDOUT:     %.loc11_5.2: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst %.loc11_5.6 [template = %.loc11_5.8 (constants.%.432)]
// CHECK:STDOUT:     %.loc11_5.3: @F.%.loc11_5.7 (@F.%.loc11_5.7) = splice_inst %.loc11_5.9 [template = %.loc11_5.10 (constants.%.263)]
// CHECK:STDOUT:     %.loc11_3: ref <error> = deref %.loc11_5.3 [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc6_25.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc6_25.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_unimplemented_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template]
// CHECK:STDOUT:   %c: %C = symbolic_binding c, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.26f: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst @F.%.loc16_5.3 [template]
// CHECK:STDOUT:   %require_complete.519: <witness> = require_complete_type @F.%.loc16_5.4 [template]
// CHECK:STDOUT:   %.2d8: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst @F.%.loc16_5.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt.loc11_16.1: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_16.2 (constants.%c.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %C.ref [concrete = constants.%C] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.loc11_16.2: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %field_decl: %C.elem = field_decl n, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%c.loc11_16.2: %C) {
// CHECK:STDOUT:   %c.patt.loc11_16.2: %pattern_type.98b = symbolic_binding_pattern c, 0, template [template = %c.patt.loc11_16.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.loc11_16.1: %C = symbolic_binding c, 0, template [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc16_5.3: <instruction> = access_member_action %c.ref, n [template]
// CHECK:STDOUT:   %.loc16_5.4: type = type_of_inst %.loc16_5.3 [template]
// CHECK:STDOUT:   %.loc16_5.5: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst %.loc16_5.3 [template = %.loc16_5.5 (constants.%.26f)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc16_5.4 [template = %require_complete (constants.%require_complete.519)]
// CHECK:STDOUT:   %.loc16_5.6: <instruction> = convert_to_category_action %.loc16_5.1, element1 [template]
// CHECK:STDOUT:   %.loc16_5.7: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst %.loc16_5.6 [template = %.loc16_5.7 (constants.%.2d8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: %C = name_ref c, %c.loc11_16.2 [template = %c.loc11_16.1 (constants.%c)]
// CHECK:STDOUT:     %.loc16_5.1: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst %.loc16_5.3 [template = %.loc16_5.5 (constants.%.26f)]
// CHECK:STDOUT:     %.loc16_5.2: @F.%.loc16_5.4 (@F.%.loc16_5.4) = splice_inst %.loc16_5.6 [template = %.loc16_5.7 (constants.%.2d8)]
// CHECK:STDOUT:     %.loc16_3: ref <error> = deref %.loc16_5.2 [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%c) {
// CHECK:STDOUT:   %c.patt.loc11_16.2 => constants.%c.patt
// CHECK:STDOUT:   %c.loc11_16.1 => constants.%c
// CHECK:STDOUT: }
// CHECK:STDOUT:
