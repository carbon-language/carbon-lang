// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/operator.carbon

// --- assign.carbon
library "[[@TEST_NAME]]";

class C {
  var x: i32;
}

//@dump-sem-ir-begin
fn F[template T: type](ref t: T) {
  t.x = 456;
}
//@dump-sem-ir-end

fn G() {
  var c: C = {.x = 123};
  F(ref c);
}

// --- todo_fail_assign_const.carbon
library "[[@TEST_NAME]]";

class C {
  var x: i32;
}

fn F[template T: type](ref t: T) {
  t.x = 456;
}

fn G() {
  var c: const C = {.x = 123};
  F(ref c);
}

// --- assign_static_var.carbon
library "[[@TEST_NAME]]";

fn F(template T: type) {
  T.x = 0;
}

class A {
  static var x: i32 = 0;
}

fn G() {
  F(A);
}

// --- todo_fail_assign_let.carbon
library "[[@TEST_NAME]]";

fn F(template T: type) {
  T.x = 0;
}

class B {
  let x: i32 = 0;
}

fn G() {
  F(B);
}

// CHECK:STDOUT: --- assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.a15: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0c6: <witness> = complete_type_witness %struct_type.x.a15 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47011.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %t.param_patt.d4a: %pattern_type.51d1c4.1 = ref_param_pattern [template]
// CHECK:STDOUT:   %t.patt.e90: %pattern_type.51d1c4.1 = wrapper_binding_pattern t, %t.param_patt.d4a [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %.bc7: type = type_of_inst @F.%.loc9_4.4 [template]
// CHECK:STDOUT:   %.ec2: %.bc7 = splice_inst @F.%.loc9_4.4 [template]
// CHECK:STDOUT:   %int_456: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %require_complete.03f: <witness> = require_complete_type %.bc7 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.740: %.bc7 = splice_inst @F.%.loc9_4.7 [template]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %t.param_patt.df1: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.e50: %pattern_type.98b = wrapper_binding_pattern t, %t.param_patt.df1 [concrete]
// CHECK:STDOUT:   %inst.splice_block.f32: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d44: ref %i32 = splice_block %.3d2 {
// CHECK:STDOUT:       %.b6d: ref %C = specific_inst @F.%t.ref, @F(%C)
// CHECK:STDOUT:       %x.ref: %C.elem = name_ref x, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.3d2: ref %i32 = class_element_access %.b6d, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.5c8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.30b: ref %i32 = splice_block %.d44 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:     %t.param_patt.loc8_29.1: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_param_pattern [template = %t.param_patt.loc8_29.2 (constants.%t.param_patt.d4a)]
// CHECK:STDOUT:     %t.patt.loc8_29.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern t, %t.param_patt.loc8_29.1 [template = %t.patt.loc8_29.2 (constants.%t.patt.e90)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %t.param: ref @F.%T.loc8_16.1 (%T.67db0b.1) = ref_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_16.2 [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %t: ref @F.%T.loc8_16.1 (%T.67db0b.1) = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %t.param_patt.loc8_29.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_param_pattern [template = %t.param_patt.loc8_29.2 (constants.%t.param_patt.d4a)]
// CHECK:STDOUT:   %t.patt.loc8_29.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern t, %t.param_patt.loc8_29.2 [template = %t.patt.loc8_29.2 (constants.%t.patt.e90)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_16.1 [template = %require_complete.loc8 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc9_4.4: <instruction> = access_member_action %t.ref, x [template]
// CHECK:STDOUT:   %.loc9_4.5: type = type_of_inst %.loc9_4.4 [template = %.loc9_4.5 (constants.%.bc7)]
// CHECK:STDOUT:   %.loc9_4.6: @F.%.loc9_4.5 (%.bc7) = splice_inst %.loc9_4.4 [template = %.loc9_4.6 (constants.%.ec2)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %.loc9_4.5 [template = %require_complete.loc9 (constants.%require_complete.03f)]
// CHECK:STDOUT:   %.loc9_7: %tuple.type = initialize_action %int_456, %.loc9_4.5, %.loc9_4.6, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_7, element0 [template]
// CHECK:STDOUT:   %.loc9_4.7: <instruction> = convert_to_category_action %.loc9_4.2, element9 [template]
// CHECK:STDOUT:   %.loc9_4.8: @F.%.loc9_4.5 (%.bc7) = splice_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.740)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc8_16.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc8_16.1 (%T.67db0b.1) = name_ref t, %t
// CHECK:STDOUT:     %.loc9_4.1: type = type_of_inst %.loc9_4.4 [template = %.loc9_4.5 (constants.%.bc7)]
// CHECK:STDOUT:     %.loc9_4.2: @F.%.loc9_4.5 (%.bc7) = splice_inst %.loc9_4.4 [template = %.loc9_4.6 (constants.%.ec2)]
// CHECK:STDOUT:     %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456]
// CHECK:STDOUT:     assign %.loc9_4.2, %tuple.elem0
// CHECK:STDOUT:     %.loc9_4.3: @F.%.loc9_4.5 (%.bc7) = splice_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.740)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %t.param_patt.loc8_29.2 => constants.%t.param_patt.d4a
// CHECK:STDOUT:   %t.patt.loc8_29.2 => constants.%t.patt.e90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %t.param_patt.loc8_29.2 => constants.%t.param_patt.df1
// CHECK:STDOUT:   %t.patt.loc8_29.2 => constants.%t.patt.e50
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.0c6
// CHECK:STDOUT:   %.loc9_4.4 => constants.%inst.splice_block.f32
// CHECK:STDOUT:   %.loc9_4.5 => constants.%i32
// CHECK:STDOUT:   %.loc9_4.6 => invalid
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc9_7 => invalid
// CHECK:STDOUT:   %tuple.elem0 => invalid
// CHECK:STDOUT:   %.loc9_4.7 => constants.%inst.splice_block.5c8
// CHECK:STDOUT:   %.loc9_4.8 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
