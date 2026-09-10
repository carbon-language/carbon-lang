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
// CHECK:STDOUT:   %.1c7: %T.67db0b.1 = splice_inst @F.%.loc9_4.5 [template]
// CHECK:STDOUT:   %.7b5: type = type_of_inst @F.%.loc9_4.7 [template]
// CHECK:STDOUT:   %.3a0: %.7b5 = splice_inst @F.%.loc9_4.7 [template]
// CHECK:STDOUT:   %int_456.010: Core.IntLiteral = int_value 456 [concrete]
// CHECK:STDOUT:   %require_complete.32d: <witness> = require_complete_type %.7b5 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa489.1: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.29426c.1: %ImplicitAs.type.3aa489.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fccd30.1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97ed63.1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.29426c.1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.9453b4.1: %ImplicitAs.WithSelf.Convert.type.97ed63.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.e74: type = type_of_inst @F.%.loc9_7.12 [template]
// CHECK:STDOUT:   %.5ed: %.e74 = splice_inst @F.%.loc9_7.12 [template]
// CHECK:STDOUT:   %.a65: type = type_of_inst @F.%.loc9_7.15 [template]
// CHECK:STDOUT:   %.482: %.a65 = splice_inst @F.%.loc9_7.15 [template]
// CHECK:STDOUT:   %.5d8: type = type_of_inst @F.%.loc9_7.18 [template]
// CHECK:STDOUT:   %.bab: %.5d8 = splice_inst @F.%.loc9_7.18 [template]
// CHECK:STDOUT:   %.6cc: type = type_of_inst @F.%.loc9_7.21 [template]
// CHECK:STDOUT:   %.e38: %.6cc = splice_inst @F.%.loc9_7.21 [template]
// CHECK:STDOUT:   %.d37: %.7b5 = splice_inst @F.%.loc9_7.24 [template]
// CHECK:STDOUT:   %.0d9: %.7b5 = splice_inst @F.%.loc9_7.26 [template]
// CHECK:STDOUT:   %.f2e: %.7b5 = splice_inst @F.%.loc9_4.10 [template]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914edc.1: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.368: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %assoc0.0ac: %ImplicitAs.assoc_type.fccd30.1 = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914edc.1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.57149a.1: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %t.param_patt.df1: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.e50: %pattern_type.98b = wrapper_binding_pattern t, %t.param_patt.df1 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.dbe: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b6d: ref %C = specific_inst @F.%t.ref, @F(%C)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.00c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.be2: ref %i32 = splice_block %.3d2 {
// CHECK:STDOUT:       %x.ref: %C.elem = name_ref x, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.3d2: ref %i32 = class_element_access %.b6d, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.facet_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %ImplicitAs.type.914edc.2: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete = %ImplicitAs.type.914edc.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.f55: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a54: %ImplicitAs.assoc_type.e23 = splice_block %Convert.ref [concrete = %assoc0.368] {
// CHECK:STDOUT:       %.88d: %ImplicitAs.assoc_type.e23 = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(%i32, %Self.29426c.1) [concrete = %assoc0.368]
// CHECK:STDOUT:       %Convert.ref: %ImplicitAs.assoc_type.e23 = name_ref Convert, %.88d [concrete = %assoc0.368]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %inst.splice_block.3b1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.895: <bound method> = splice_block %bound_method.6bb [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc] {
// CHECK:STDOUT:       %impl.elem0.0e6: %.b7a = impl_witness_access %ImplicitAs.impl_witness.a2a, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.6bb: <bound method> = bound_method %int_456.010, %impl.elem0.0e6 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bound_method.c5c699.1: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.57149a.1 [concrete]
// CHECK:STDOUT:   %int_456.8dd: %i32 = int_value 456 [concrete]
// CHECK:STDOUT:   %inst.splice_block.e9c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.367: init %i32 = splice_block %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = %int_456.8dd] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %bound_method.c5c699.2: <bound method> = bound_method %int_456.010, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.57149a.2 [concrete = %bound_method.c5c699.1]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.c5c699.2(%int_456.010) [concrete = %int_456.8dd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.45a: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.8f7: %i32 = specific_inst %int_456.8dd, @F(%C) [concrete = %int_456.8dd]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.b29: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c5c: init %i32 = splice_block %Int.as.Copy.impl.Op.call [concrete = %int_456.8dd] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.d8a: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d78: ref %i32 = splice_block %.be2 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fccd30.1) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac)]
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97ed63.1) = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.9453b4.1)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//prelude/operators/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
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
// CHECK:STDOUT:   %.loc9_4.5: <instruction> = refine_inst_action %t.ref [template]
// CHECK:STDOUT:   %.loc9_4.6: @F.%T.loc8_16.1 (%T.67db0b.1) = splice_inst %.loc9_4.5 [template = %.loc9_4.6 (constants.%.1c7)]
// CHECK:STDOUT:   %.loc9_4.7: <instruction> = access_member_action %.loc9_4.1, x [template]
// CHECK:STDOUT:   %.loc9_4.8: type = type_of_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.7b5)]
// CHECK:STDOUT:   %.loc9_4.9: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_4.7 [template = %.loc9_4.9 (constants.%.3a0)]
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %.loc9_4.8 [template = %require_complete.loc9 (constants.%require_complete.32d)]
// CHECK:STDOUT:   %.loc9_7.12: <instruction> = call_action (constants.%ImplicitAs.generic, %.loc9_4.8), false [template]
// CHECK:STDOUT:   %.loc9_7.13: type = type_of_inst %.loc9_7.12 [template = %.loc9_7.13 (constants.%.e74)]
// CHECK:STDOUT:   %.loc9_7.14: @F.%.loc9_7.13 (%.e74) = splice_inst %.loc9_7.12 [template = %.loc9_7.14 (constants.%.5ed)]
// CHECK:STDOUT:   %.loc9_7.15: <instruction> = access_member_action %.loc9_7.2, Convert [template]
// CHECK:STDOUT:   %.loc9_7.16: type = type_of_inst %.loc9_7.15 [template = %.loc9_7.16 (constants.%.a65)]
// CHECK:STDOUT:   %.loc9_7.17: @F.%.loc9_7.16 (%.a65) = splice_inst %.loc9_7.15 [template = %.loc9_7.17 (constants.%.482)]
// CHECK:STDOUT:   %.loc9_7.18: <instruction> = compound_member_access_action %int_456, %.loc9_7.4 [template]
// CHECK:STDOUT:   %.loc9_7.19: type = type_of_inst %.loc9_7.18 [template = %.loc9_7.19 (constants.%.5d8)]
// CHECK:STDOUT:   %.loc9_7.20: @F.%.loc9_7.19 (%.5d8) = splice_inst %.loc9_7.18 [template = %.loc9_7.20 (constants.%.bab)]
// CHECK:STDOUT:   %.loc9_7.21: <instruction> = call_action (%.loc9_7.6), true [template]
// CHECK:STDOUT:   %.loc9_7.22: type = type_of_inst %.loc9_7.21 [template = %.loc9_7.22 (constants.%.6cc)]
// CHECK:STDOUT:   %.loc9_7.23: @F.%.loc9_7.22 (%.6cc) = splice_inst %.loc9_7.21 [template = %.loc9_7.23 (constants.%.e38)]
// CHECK:STDOUT:   %.loc9_7.24: <instruction> = refine_inst_action %.loc9_7.9 [template]
// CHECK:STDOUT:   %.loc9_7.25: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_7.24 [template = %.loc9_7.25 (constants.%.d37)]
// CHECK:STDOUT:   %.loc9_7.26: <instruction> = convert_to_category_action %.loc9_7.10, element10 [template]
// CHECK:STDOUT:   %.loc9_7.27: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_7.26 [template = %.loc9_7.27 (constants.%.0d9)]
// CHECK:STDOUT:   %.loc9_4.10: <instruction> = convert_to_category_action %.loc9_4.3, element9 [template]
// CHECK:STDOUT:   %.loc9_4.11: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_4.10 [template = %.loc9_4.11 (constants.%.f2e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.loc8_16.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.loc8_16.1 (%T.67db0b.1) = name_ref t, %t
// CHECK:STDOUT:     %.loc9_4.1: @F.%T.loc8_16.1 (%T.67db0b.1) = splice_inst %.loc9_4.5 [template = %.loc9_4.6 (constants.%.1c7)]
// CHECK:STDOUT:     %.loc9_4.2: type = type_of_inst %.loc9_4.7 [template = %.loc9_4.8 (constants.%.7b5)]
// CHECK:STDOUT:     %.loc9_4.3: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_4.7 [template = %.loc9_4.9 (constants.%.3a0)]
// CHECK:STDOUT:     %int_456: Core.IntLiteral = int_value 456 [concrete = constants.%int_456.010]
// CHECK:STDOUT:     %.loc9_7.1: type = type_of_inst %.loc9_7.12 [template = %.loc9_7.13 (constants.%.e74)]
// CHECK:STDOUT:     %.loc9_7.2: @F.%.loc9_7.13 (%.e74) = splice_inst %.loc9_7.12 [template = %.loc9_7.14 (constants.%.5ed)]
// CHECK:STDOUT:     %.loc9_7.3: type = type_of_inst %.loc9_7.15 [template = %.loc9_7.16 (constants.%.a65)]
// CHECK:STDOUT:     %.loc9_7.4: @F.%.loc9_7.16 (%.a65) = splice_inst %.loc9_7.15 [template = %.loc9_7.17 (constants.%.482)]
// CHECK:STDOUT:     %.loc9_7.5: type = type_of_inst %.loc9_7.18 [template = %.loc9_7.19 (constants.%.5d8)]
// CHECK:STDOUT:     %.loc9_7.6: @F.%.loc9_7.19 (%.5d8) = splice_inst %.loc9_7.18 [template = %.loc9_7.20 (constants.%.bab)]
// CHECK:STDOUT:     %.loc9_7.7: type = type_of_inst %.loc9_7.21 [template = %.loc9_7.22 (constants.%.6cc)]
// CHECK:STDOUT:     %.loc9_7.8: @F.%.loc9_7.22 (%.6cc) = splice_inst %.loc9_7.21 [template = %.loc9_7.23 (constants.%.e38)]
// CHECK:STDOUT:     %.loc9_7.9: @F.%.loc9_4.8 (%.7b5) = converted %int_456, %.loc9_7.8 [template = %.loc9_7.23 (constants.%.e38)]
// CHECK:STDOUT:     %.loc9_7.10: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_7.24 [template = %.loc9_7.25 (constants.%.d37)]
// CHECK:STDOUT:     %.loc9_7.11: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_7.26 [template = %.loc9_7.27 (constants.%.0d9)]
// CHECK:STDOUT:     assign %.loc9_4.3, %.loc9_7.11
// CHECK:STDOUT:     %.loc9_4.4: @F.%.loc9_4.8 (%.7b5) = splice_inst %.loc9_4.10 [template = %.loc9_4.11 (constants.%.f2e)]
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
// CHECK:STDOUT:   %.loc9_4.5 => constants.%inst.specific_inst.dbe
// CHECK:STDOUT:   %.loc9_4.6 => invalid
// CHECK:STDOUT:   %.loc9_4.7 => constants.%inst.splice_block.00c
// CHECK:STDOUT:   %.loc9_4.8 => constants.%i32
// CHECK:STDOUT:   %.loc9_4.9 => invalid
// CHECK:STDOUT:   %require_complete.loc9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc9_7.12 => constants.%inst.facet_type
// CHECK:STDOUT:   %.loc9_7.13 => type
// CHECK:STDOUT:   %.loc9_7.14 => constants.%ImplicitAs.type.914edc.1
// CHECK:STDOUT:   %.loc9_7.15 => constants.%inst.splice_block.f55
// CHECK:STDOUT:   %.loc9_7.16 => constants.%ImplicitAs.assoc_type.e23
// CHECK:STDOUT:   %.loc9_7.17 => constants.%assoc0.368
// CHECK:STDOUT:   %.loc9_7.18 => constants.%inst.splice_block.3b1
// CHECK:STDOUT:   %.loc9_7.19 => <bound method>
// CHECK:STDOUT:   %.loc9_7.20 => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.efc
// CHECK:STDOUT:   %.loc9_7.21 => constants.%inst.splice_block.e9c
// CHECK:STDOUT:   %.loc9_7.22 => constants.%i32
// CHECK:STDOUT:   %.loc9_7.23 => constants.%int_456.8dd
// CHECK:STDOUT:   %.loc9_7.24 => constants.%inst.specific_inst.45a
// CHECK:STDOUT:   %.loc9_7.25 => constants.%int_456.8dd
// CHECK:STDOUT:   %.loc9_7.26 => constants.%inst.splice_block.b29
// CHECK:STDOUT:   %.loc9_7.27 => constants.%int_456.8dd
// CHECK:STDOUT:   %.loc9_4.10 => constants.%inst.splice_block.d8a
// CHECK:STDOUT:   %.loc9_4.11 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
