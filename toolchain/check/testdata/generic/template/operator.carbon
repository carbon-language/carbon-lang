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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.fb6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a7b: <witness> = complete_type_witness %struct_type.x.fb6 [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %t.param_patt.d4a: %pattern_type.51d1c4.1 = ref_param_pattern [template]
// CHECK:STDOUT:   %t.patt.e90: %pattern_type.51d1c4.1 = wrapper_binding_pattern t, %t.param_patt.d4a [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %.939: type = type_of_inst @F.%.loc9_4.4 [template]
// CHECK:STDOUT:   %.fa5: %.939 = splice_inst @F.%.loc9_4.4 [template]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %require_complete.906: <witness> = require_complete_type %.939 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.6a7: %.939 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %.cba: %.939 = splice_inst @F.%.loc9_4.7 [template]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %t.param_patt.df1: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.e50: %pattern_type.98b = wrapper_binding_pattern t, %t.param_patt.df1 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.8d8: ref %i32 = splice_block %.25b {
// CHECK:STDOUT:       %.b6d: ref %C = specific_inst @F.%t.ref, @F(%C)
// CHECK:STDOUT:       %x.ref: %C.elem = name_ref x, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.25b: ref %i32 = class_element_access %.b6d, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.f3a: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_456.3b8: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fca: init %i32 = splice_block %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = %int_456.3b8] {
// CHECK:STDOUT:       %impl.elem0.5e5: %.208 = impl_witness_access %ImplicitAs.impl_witness.e87, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:       %bound_method.75f: <bound method> = bound_method %int_456.010, %impl.elem0.5e5 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.afb]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.5e5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.74e: <bound method> = bound_method %int_456.010, %specific_fn [concrete = %bound_method.f3a]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.74e(%int_456.010) [concrete = %int_456.3b8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.2) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b4d: ref %i32 = splice_block %.8d8 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %t.param_patt.loc8_29.1: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_param_pattern [template = %t.param_patt.loc8_29.2 (constants.%t.param_patt.d4a)]
// CHECK:STDOUT:     %t.patt.loc8_29.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern t, %t.param_patt.loc8_29.1 [template = %t.patt.loc8_29.2 (constants.%t.patt.e90)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
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
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %t.param_patt.loc8_29.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_param_pattern [template = %t.param_patt.loc8_29.2 (constants.%t.param_patt.d4a)]
// CHECK:STDOUT:   %t.patt.loc8_29.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern t, %t.param_patt.loc8_29.2 [template = %t.patt.loc8_29.2 (constants.%t.patt.e90)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc8_16.1 [template = %require_complete.loc8 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc9_4.4: <instruction> = access_member_action %t.ref, x [template]
// CHECK:STDOUT:   %.loc9_4.5: type = type_of_inst %.loc9_4.4 [template = %.loc9_4.5 (constants.%.939)]
// CHECK:STDOUT:   %.loc9_4.6: @F.%.loc9_4.5 (%.939) = splice_inst %.loc9_4.4 [template = %.loc9_4.6 (constants.%.fa5)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %.loc9_4.5 [template = %require_complete.loc9 (constants.%require_complete.906)]
// CHECK:STDOUT:   %.loc9_7.2: %tuple.type = initialize_action %int_456, %.loc9_4.5, %.loc9_4.2, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_7.2, element0 [template]
// CHECK:STDOUT:   %.loc9_7.3: @F.%.loc9_4.5 (%.939) = splice_inst %tuple.elem0 [template = %.loc9_7.3 (constants.%.6a7)]
// CHECK:STDOUT:   %.loc9_4.7: <instruction> = convert_to_category_action %.loc9_4.2, element9 [template]
// CHECK:STDOUT:   %.loc9_4.8: @F.%.loc9_4.5 (%.939) = splice_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.cba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc8_16.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc8_16.1 (%T.67db0b.1) = name_ref t, %t
// CHECK:STDOUT:     %.loc9_4.1: type = type_of_inst %.loc9_4.4 [template = %.loc9_4.5 (constants.%.939)]
// CHECK:STDOUT:     %.loc9_4.2: @F.%.loc9_4.5 (%.939) = splice_inst %.loc9_4.4 [template = %.loc9_4.6 (constants.%.fa5)]
// CHECK:STDOUT:     %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:     %.loc9_7.1: @F.%.loc9_4.5 (%.939) = splice_inst %tuple.elem0 [template = %.loc9_7.3 (constants.%.6a7)]
// CHECK:STDOUT:     assign %.loc9_4.2, %.loc9_7.1
// CHECK:STDOUT:     %.loc9_4.3: @F.%.loc9_4.5 (%.939) = splice_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.cba)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %t.param_patt.loc8_29.2 => constants.%t.param_patt.d4a
// CHECK:STDOUT:   %t.patt.loc8_29.2 => constants.%t.patt.e90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %t.param_patt.loc8_29.2 => constants.%t.param_patt.df1
// CHECK:STDOUT:   %t.patt.loc8_29.2 => constants.%t.patt.e50
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.a7b
// CHECK:STDOUT:   %.loc9_4.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc9_4.5 => constants.%i32
// CHECK:STDOUT:   %.loc9_4.6 => <not constant>
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc9_7.2 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_7.3 => constants.%int_456.3b8
// CHECK:STDOUT:   %.loc9_4.7 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_4.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
