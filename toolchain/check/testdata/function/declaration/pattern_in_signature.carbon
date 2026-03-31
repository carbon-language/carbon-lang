// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/pattern_in_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/pattern_in_signature.carbon

//@dump-sem-ir-begin
fn F((unused a: {}, unused b: {}), unused c: {}) {}
//@dump-sem-ir-end

//@dump-sem-ir-begin
fn CallF(ab: ({}, {}), c: {}) {
  F(ab, c);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- pattern_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.de4: type = pattern_type %tuple.type.b6b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %CallF.type: type = fn_type @CallF [concrete]
// CHECK:STDOUT:   %CallF: %CallF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.a96 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.a96 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %.loc14_33: %pattern_type.de4 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.a96 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc14_18.1: type = splice_block %.loc14_18.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc14_18.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_18.3: type = converted %.loc14_18.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %empty_struct_type = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc14_32.1: type = splice_block %.loc14_32.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc14_32.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_32.3: type = converted %.loc14_32.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %empty_struct_type = value_binding b, %b.param
// CHECK:STDOUT:     %c.param: %empty_struct_type = value_param call_param2
// CHECK:STDOUT:     %.loc14_47.1: type = splice_block %.loc14_47.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc14_47.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc14_47.3: type = converted %.loc14_47.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %empty_struct_type = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallF.decl: %CallF.type = fn_decl @CallF [concrete = constants.%CallF] {
// CHECK:STDOUT:     %ab.param_patt: %pattern_type.de4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %ab.patt: %pattern_type.de4 = at_binding_pattern ab, %ab.param_patt [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.a96 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.a96 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ab.param: %tuple.type.b6b = value_param call_param0
// CHECK:STDOUT:     %.loc18_21.1: type = splice_block %.loc18_21.5 [concrete = constants.%tuple.type.b6b] {
// CHECK:STDOUT:       %.loc18_16: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc18_20: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc18_21.2: %tuple.type.b6b = tuple_literal (%.loc18_16, %.loc18_20) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc18_21.3: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc18_21.4: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc18_21.5: type = converted %.loc18_21.2, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ab: %tuple.type.b6b = value_binding ab, %ab.param
// CHECK:STDOUT:     %c.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc18_28.1: type = splice_block %.loc18_28.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc18_28.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc18_28.3: type = converted %.loc18_28.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %empty_struct_type = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %empty_struct_type, %b.param: %empty_struct_type, %c.param: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallF(%ab.param: %tuple.type.b6b, %c.param: %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %ab.ref: %tuple.type.b6b = name_ref ab, %ab
// CHECK:STDOUT:   %c.ref: %empty_struct_type = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0: %empty_struct_type = tuple_access %ab.ref, element0
// CHECK:STDOUT:   %tuple.elem1: %empty_struct_type = tuple_access %ab.ref, element1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%tuple.elem0, %tuple.elem1, %c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
