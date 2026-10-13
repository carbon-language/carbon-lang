// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/local_function.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn Test(unused generic T: type) {
  fn Local() {
  }
  Local();
}
//@dump-sem-ir-end

fn CallTest() {
  Test(i32);
}

// --- use_enclosing_local.carbon
library "[[@TEST_NAME]]";

class S {
  fn Make() -> S { return {.n = 42}; }
  var n: i32;
}

fn EnclosingTemplate(template T: type) -> i32 {
  let a: T = T.Make();
  // We can refer to `a.n` here even though that's a template-dependent value in
  // an enclosing generic.
  //@dump-sem-ir-begin
  fn Local(_: T, y: typeof(a.n)) -> typeof(a.n) {
    return y;
  }
  //@dump-sem-ir-end
  return Local(T.Make(), a.n);
}

fn CallEnclosingTemplate() -> i32 {
  return EnclosingTemplate(S);
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %Local.type.c16: type = fn_type @Local, @Test(%T) [symbolic]
// CHECK:STDOUT:   %Local.76d: %Local.type.c16 = struct_value () [symbolic]
// CHECK:STDOUT:   %Local.specific_fn.145: <specific function> = specific_function %Local.76d, @Local(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Local.type.2aa: type = fn_type @Local, @Test(%i32) [concrete]
// CHECK:STDOUT:   %Local.84c: %Local.type.2aa = struct_value () [concrete]
// CHECK:STDOUT:   %Local.specific_fn.383: <specific function> = specific_function %Local.84c, @Local(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %T.patt.loc12_25.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_27.1: type = splice_block %.loc12_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc12_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_25.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_25.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc12_25.2: type) {
// CHECK:STDOUT:   %T.patt.loc12_25.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc12_25.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_25.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Local.type: type = fn_type @Local, @Test(%T.loc12_25.1) [symbolic = %Local.type (constants.%Local.type.c16)]
// CHECK:STDOUT:   %Local: @Test.%Local.type (%Local.type.c16) = struct_value () [symbolic = %Local (constants.%Local.76d)]
// CHECK:STDOUT:   %Local.specific_fn.loc15_3.2: <specific function> = specific_function %Local, @Local(%T.loc12_25.1) [symbolic = %Local.specific_fn.loc15_3.2 (constants.%Local.specific_fn.145)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Local.decl: @Test.%Local.type (%Local.type.c16) = fn_decl @Local [symbolic = @Test.%Local (constants.%Local.76d)] {} {}
// CHECK:STDOUT:     %Local.ref: @Test.%Local.type (%Local.type.c16) = name_ref Local, %Local.decl [symbolic = %Local (constants.%Local.76d)]
// CHECK:STDOUT:     %Local.specific_fn.loc15_3.1: <specific function> = specific_function %Local.ref, @Local(constants.%T) [symbolic = %Local.specific_fn.loc15_3.2 (constants.%Local.specific_fn.145)]
// CHECK:STDOUT:     %Local.call: init %empty_tuple.type = call %Local.specific_fn.loc15_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Local(@Test.%T.loc12_25.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc12_25.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_25.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc12_25.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_25.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Local.type => constants.%Local.type.2aa
// CHECK:STDOUT:   %Local => constants.%Local.84c
// CHECK:STDOUT:   %Local.specific_fn.loc15_3.2 => constants.%Local.specific_fn.383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_enclosing_local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %pattern_type.a55: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.Make.type: type = fn_type @S.Make [concrete]
// CHECK:STDOUT:   %S.Make: %S.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %S.elem: type = unbound_element_type %S, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.9ed: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n.9ed [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %_.param_patt.339: %pattern_type.51d1c4.2 = value_param_pattern [template]
// CHECK:STDOUT:   %_.patt.ace: %pattern_type.51d1c4.2 = wrapper_binding_pattern _, %_.param_patt.339 [template]
// CHECK:STDOUT:   %.237907.1: type = type_of_inst @Local.%.loc13_29.1 [template]
// CHECK:STDOUT:   %.f9658d.1: %.237907.1 = splice_inst @Local.%.loc13_29.1 [template]
// CHECK:STDOUT:   %pattern_type.940805.1: type = pattern_type %.237907.1 [template]
// CHECK:STDOUT:   %y.param_patt.8246e7.1: %pattern_type.940805.1 = value_param_pattern [template]
// CHECK:STDOUT:   %y.patt.d967f4.1: %pattern_type.940805.1 = wrapper_binding_pattern y, %y.param_patt.8246e7.1 [template]
// CHECK:STDOUT:   %.237907.2: type = type_of_inst @Local.%.loc13_45.1 [template]
// CHECK:STDOUT:   %.f9658d.2: %.237907.2 = splice_inst @Local.%.loc13_45.1 [template]
// CHECK:STDOUT:   %.6c4209.1: Core.Form = init_form %.237907.2 [template]
// CHECK:STDOUT:   %pattern_type.940805.2: type = pattern_type %.237907.2 [template]
// CHECK:STDOUT:   %return.param_patt.bcf038.1: %pattern_type.940805.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.690db9.1: %pattern_type.940805.2 = return_slot_pattern %return.param_patt.bcf038.1, %.237907.2 [template]
// CHECK:STDOUT:   %.e9daaa.1: <instruction> = access_member_action @Local.%a.ref.loc13_28, n [template]
// CHECK:STDOUT:   %.237907.3: type = type_of_inst %.e9daaa.1 [template]
// CHECK:STDOUT:   %.f9658d.3: %.237907.3 = splice_inst %.e9daaa.1 [template]
// CHECK:STDOUT:   %pattern_type.940805.3: type = pattern_type %.237907.3 [template]
// CHECK:STDOUT:   %y.param_patt.8246e7.2: %pattern_type.940805.3 = value_param_pattern [template]
// CHECK:STDOUT:   %y.patt.d967f4.2: %pattern_type.940805.3 = wrapper_binding_pattern y, %y.param_patt.8246e7.2 [template]
// CHECK:STDOUT:   %.e9daaa.2: <instruction> = access_member_action @Local.%a.ref.loc13_44, n [template]
// CHECK:STDOUT:   %.237907.4: type = type_of_inst %.e9daaa.2 [template]
// CHECK:STDOUT:   %.f9658d.4: %.237907.4 = splice_inst %.e9daaa.2 [template]
// CHECK:STDOUT:   %.6c4209.2: Core.Form = init_form %.237907.4 [template]
// CHECK:STDOUT:   %pattern_type.940805.4: type = pattern_type %.237907.4 [template]
// CHECK:STDOUT:   %return.param_patt.bcf038.2: %pattern_type.940805.4 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.690db9.2: %pattern_type.940805.4 = return_slot_pattern %return.param_patt.bcf038.2, %.237907.4 [template]
// CHECK:STDOUT:   %Local.type.348: type = fn_type @Local, @EnclosingTemplate(%T.67db0b.1) [template]
// CHECK:STDOUT:   %Local.cb2: %Local.type.348 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.b43ae1.1: <witness> = require_complete_type %.237907.1 [template]
// CHECK:STDOUT:   %require_complete.b43ae1.2: <witness> = require_complete_type %.237907.2 [template]
// CHECK:STDOUT:   %tuple.type.f7a: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.26b: %.237907.2 = splice_inst @Local.%tuple.elem0 [template]
// CHECK:STDOUT:   %a.patt.2a2: %pattern_type.a55 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %inst.name_ref.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Make.ref.b85793.1: %S.Make.type = name_ref Make, @S.%S.Make.decl [concrete = %S.Make]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.058268.1: init %S = splice_block %S.Make.call.2c2051.2 {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.349: %S = splice_block %.ff4281.2 {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Local.type.2cd: type = fn_type @Local, @EnclosingTemplate(%S) [concrete]
// CHECK:STDOUT:   %Local.6cd: %Local.type.2cd = struct_value () [concrete]
// CHECK:STDOUT:   %inst.name_ref.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Make.ref.b85793.2: %S.Make.type = name_ref Make, @S.%S.Make.decl [concrete = %S.Make]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.058268.2: init %S = splice_block %S.Make.call.2c2051.1 {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7eb: %i32 = splice_block %.5ea {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.param_patt.230: %pattern_type.a55 = value_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.e4c: %pattern_type.a55 = wrapper_binding_pattern _, %_.param_patt.230 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.591303.1: %i32 = splice_block %.dcb12d.2 {
// CHECK:STDOUT:       %.ac15a1.2: %S = specific_inst @Local.%a.ref.loc13_28, @Local(%S)
// CHECK:STDOUT:       %n.ref.1bbe5e.2: %S.elem = name_ref n, @S.%field_decl [concrete = @S.%field_decl]
// CHECK:STDOUT:       %.817716.2: ref %i32 = class_element_access %.ac15a1.2, element0
// CHECK:STDOUT:       %.dcb12d.2: %i32 = acquire_value %.817716.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.param_patt.673: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.547: %pattern_type.6b6 = wrapper_binding_pattern y, %y.param_patt.673 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.591303.2: %i32 = splice_block %.dcb12d.1 {
// CHECK:STDOUT:       %.ac15a1.1: %S = specific_inst @Local.%a.ref.loc13_44, @Local(%S)
// CHECK:STDOUT:       %n.ref.1bbe5e.1: %S.elem = name_ref n, @S.%field_decl [concrete = @S.%field_decl]
// CHECK:STDOUT:       %.817716.1: ref %i32 = class_element_access %.ac15a1.1, element0
// CHECK:STDOUT:       %.dcb12d.1: %i32 = acquire_value %.817716.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.10e: init %i32 = splice_block %Local.call {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.102: init %i32 = splice_block %.10e {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.9: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.404: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %.185: %i32 = specific_inst @Local.%y.ref, @Local(%S)
// CHECK:STDOUT:       %impl.elem0.882: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.b1a: <bound method> = bound_method %.185, %impl.elem0.882
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.882, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.afd: <bound method> = bound_method %.185, %specific_fn
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.afd(%.185)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.242: %tuple.type.f7a = tuple_value (%inst.splice_block.426619.9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EnclosingTemplate(%T.loc8_32.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Local.type: type = fn_type @Local, @EnclosingTemplate(%T.loc8_32.1) [template = %Local.type (constants.%Local.type.348)]
// CHECK:STDOUT:   %Local: @EnclosingTemplate.%Local.type (%Local.type.348) = struct_value () [template = %Local (constants.%Local.cb2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Local.decl: @EnclosingTemplate.%Local.type (%Local.type.348) = fn_decl @Local [template = @EnclosingTemplate.%Local (constants.%Local.cb2)] {
// CHECK:STDOUT:       %_.param_patt.loc13_13.1: @Local.%pattern_type.loc13_13 (%pattern_type.51d1c4.2) = value_param_pattern [template = %_.param_patt.loc13_13.2 (constants.%_.param_patt.339)]
// CHECK:STDOUT:       %_.patt.loc13_13.1: @Local.%pattern_type.loc13_13 (%pattern_type.51d1c4.2) = wrapper_binding_pattern _, %_.param_patt.loc13_13.1 [template = %_.patt.loc13_13.2 (constants.%_.patt.ace)]
// CHECK:STDOUT:       %y.param_patt.loc13_19.1: @Local.%pattern_type.loc13_19 (%pattern_type.940805.1) = value_param_pattern [template = %y.param_patt.loc13_19.2 (constants.%y.param_patt.8246e7.1)]
// CHECK:STDOUT:       %y.patt.loc13_19.1: @Local.%pattern_type.loc13_19 (%pattern_type.940805.1) = wrapper_binding_pattern y, %y.param_patt.loc13_19.1 [template = %y.patt.loc13_19.2 (constants.%y.patt.d967f4.1)]
// CHECK:STDOUT:       %return.param_patt.loc13_47.1: @Local.%pattern_type.loc13_47 (%pattern_type.940805.2) = out_param_pattern [template = %return.param_patt.loc13_47.2 (constants.%return.param_patt.bcf038.1)]
// CHECK:STDOUT:       %return.patt.loc13_34.1: @Local.%pattern_type.loc13_47 (%pattern_type.940805.2) = return_slot_pattern %return.param_patt.loc13_47.1, %.loc13_47.2 [template = %return.patt.loc13_34.2 (constants.%return.patt.690db9.1)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13_47.2: type = type_of %.loc13_45.5 in [template = %.loc13_45.2 (constants.%.237907.2)] {
// CHECK:STDOUT:         %.loc13_44: @Local.%T (%T.67db0b.1) = specific_inst @EnclosingTemplate.%a, @EnclosingTemplate(constants.%T.67db0b.1)
// CHECK:STDOUT:         %a.ref.loc13_44: @Local.%T (%T.67db0b.1) = name_ref a, %.loc13_44
// CHECK:STDOUT:         %.loc13_45.4: type = type_of_inst %.loc13_45.1 [template = %.loc13_45.2 (constants.%.237907.2)]
// CHECK:STDOUT:         %.loc13_45.5: @Local.%.loc13_45.2 (%.237907.2) = splice_inst %.loc13_45.1 [template = %.loc13_45.3 (constants.%.f9658d.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.loc13_47.3: Core.Form = init_form %.loc13_47.2 [template = %.loc13_47.1 (constants.%.6c4209.1)]
// CHECK:STDOUT:       %_.param: @Local.%T (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:       %.loc13_15.1: type = splice_block %T.ref [template = %T (constants.%T.67db0b.1)] {
// CHECK:STDOUT:         %.loc13_15.2: type = specific_constant @EnclosingTemplate.%T.loc8_32.2, @EnclosingTemplate(constants.%T.67db0b.1) [template = %T (constants.%T.67db0b.1)]
// CHECK:STDOUT:         %T.ref: type = name_ref T, %.loc13_15.2 [template = %T (constants.%T.67db0b.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %_: @Local.%T (%T.67db0b.1) = wrapper_binding _, %_.param
// CHECK:STDOUT:       %y.param: @Local.%.loc13_29.2 (%.237907.1) = value_param call_param1
// CHECK:STDOUT:       %.loc13_31: type = type_of %.loc13_29.5 in [template = %.loc13_29.2 (constants.%.237907.1)] {
// CHECK:STDOUT:         %.loc13_28: @Local.%T (%T.67db0b.1) = specific_inst @EnclosingTemplate.%a, @EnclosingTemplate(constants.%T.67db0b.1)
// CHECK:STDOUT:         %a.ref.loc13_28: @Local.%T (%T.67db0b.1) = name_ref a, %.loc13_28
// CHECK:STDOUT:         %.loc13_29.4: type = type_of_inst %.loc13_29.1 [template = %.loc13_29.2 (constants.%.237907.1)]
// CHECK:STDOUT:         %.loc13_29.5: @Local.%.loc13_29.2 (%.237907.1) = splice_inst %.loc13_29.1 [template = %.loc13_29.3 (constants.%.f9658d.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @Local.%.loc13_29.2 (%.237907.1) = wrapper_binding y, %y.param
// CHECK:STDOUT:       %return.param: ref @Local.%.loc13_45.2 (%.237907.2) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Local.%.loc13_45.2 (%.237907.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Local(@EnclosingTemplate.%T.loc8_32.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type.loc13_13: type = pattern_type %T [template = %pattern_type.loc13_13 (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:   %_.param_patt.loc13_13.2: @Local.%pattern_type.loc13_13 (%pattern_type.51d1c4.2) = value_param_pattern [template = %_.param_patt.loc13_13.2 (constants.%_.param_patt.339)]
// CHECK:STDOUT:   %_.patt.loc13_13.2: @Local.%pattern_type.loc13_13 (%pattern_type.51d1c4.2) = wrapper_binding_pattern _, %_.param_patt.loc13_13.2 [template = %_.patt.loc13_13.2 (constants.%_.patt.ace)]
// CHECK:STDOUT:   %.loc13_29.1: <instruction> = access_member_action %a.ref.loc13_28, n [template]
// CHECK:STDOUT:   %.loc13_29.2: type = type_of_inst %.loc13_29.1 [template = %.loc13_29.2 (constants.%.237907.1)]
// CHECK:STDOUT:   %.loc13_29.3: @Local.%.loc13_29.2 (%.237907.1) = splice_inst %.loc13_29.1 [template = %.loc13_29.3 (constants.%.f9658d.1)]
// CHECK:STDOUT:   %pattern_type.loc13_19: type = pattern_type %.loc13_29.2 [template = %pattern_type.loc13_19 (constants.%pattern_type.940805.1)]
// CHECK:STDOUT:   %y.param_patt.loc13_19.2: @Local.%pattern_type.loc13_19 (%pattern_type.940805.1) = value_param_pattern [template = %y.param_patt.loc13_19.2 (constants.%y.param_patt.8246e7.1)]
// CHECK:STDOUT:   %y.patt.loc13_19.2: @Local.%pattern_type.loc13_19 (%pattern_type.940805.1) = wrapper_binding_pattern y, %y.param_patt.loc13_19.2 [template = %y.patt.loc13_19.2 (constants.%y.patt.d967f4.1)]
// CHECK:STDOUT:   %.loc13_45.1: <instruction> = access_member_action %a.ref.loc13_44, n [template]
// CHECK:STDOUT:   %.loc13_45.2: type = type_of_inst %.loc13_45.1 [template = %.loc13_45.2 (constants.%.237907.2)]
// CHECK:STDOUT:   %.loc13_45.3: @Local.%.loc13_45.2 (%.237907.2) = splice_inst %.loc13_45.1 [template = %.loc13_45.3 (constants.%.f9658d.2)]
// CHECK:STDOUT:   %.loc13_47.1: Core.Form = init_form %.loc13_45.2 [template = %.loc13_47.1 (constants.%.6c4209.1)]
// CHECK:STDOUT:   %pattern_type.loc13_47: type = pattern_type %.loc13_45.2 [template = %pattern_type.loc13_47 (constants.%pattern_type.940805.2)]
// CHECK:STDOUT:   %return.param_patt.loc13_47.2: @Local.%pattern_type.loc13_47 (%pattern_type.940805.2) = out_param_pattern [template = %return.param_patt.loc13_47.2 (constants.%return.param_patt.bcf038.1)]
// CHECK:STDOUT:   %return.patt.loc13_34.2: @Local.%pattern_type.loc13_47 (%pattern_type.940805.2) = return_slot_pattern %return.param_patt.loc13_47.2, %.loc13_45.2 [template = %return.patt.loc13_34.2 (constants.%return.patt.690db9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_13: <witness> = require_complete_type %T [template = %require_complete.loc13_13 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc13_19: <witness> = require_complete_type %.loc13_29.2 [template = %require_complete.loc13_19 (constants.%require_complete.b43ae1.1)]
// CHECK:STDOUT:   %require_complete.loc13_34: <witness> = require_complete_type %.loc13_45.2 [template = %require_complete.loc13_34 (constants.%require_complete.b43ae1.2)]
// CHECK:STDOUT:   %.loc14_13.2: %tuple.type.f7a = initialize_action %y.ref, %.loc13_45.2, %return.param, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc14_13.2, element0 [template]
// CHECK:STDOUT:   %.loc14_13.3: @Local.%.loc13_45.2 (%.237907.2) = splice_inst %tuple.elem0 [template = %.loc14_13.3 (constants.%.26b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: @Local.%T (%T.67db0b.1), %y.param: @Local.%.loc13_29.2 (%.237907.1)) -> out %return.param: @Local.%.loc13_45.2 (%.237907.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Local.%.loc13_29.2 (%.237907.1) = name_ref y, %y
// CHECK:STDOUT:     %.loc14_13.1: @Local.%.loc13_45.2 (%.237907.2) = splice_inst %tuple.elem0 [template = %.loc14_13.3 (constants.%.26b)]
// CHECK:STDOUT:     return %.loc14_13.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EnclosingTemplate(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc8_32.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_32.1 => constants.%T.67db0b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %_.param_patt.loc13_13.2 => constants.%_.param_patt.339
// CHECK:STDOUT:   %_.patt.loc13_13.2 => constants.%_.patt.ace
// CHECK:STDOUT:   %.loc13_29.1 => constants.%.e9daaa.1
// CHECK:STDOUT:   %.loc13_29.2 => constants.%.237907.3
// CHECK:STDOUT:   %.loc13_29.3 => constants.%.f9658d.3
// CHECK:STDOUT:   %pattern_type.loc13_19 => constants.%pattern_type.940805.3
// CHECK:STDOUT:   %y.param_patt.loc13_19.2 => constants.%y.param_patt.8246e7.2
// CHECK:STDOUT:   %y.patt.loc13_19.2 => constants.%y.patt.d967f4.2
// CHECK:STDOUT:   %.loc13_45.1 => constants.%.e9daaa.2
// CHECK:STDOUT:   %.loc13_45.2 => constants.%.237907.4
// CHECK:STDOUT:   %.loc13_45.3 => constants.%.f9658d.4
// CHECK:STDOUT:   %.loc13_47.1 => constants.%.6c4209.2
// CHECK:STDOUT:   %pattern_type.loc13_47 => constants.%pattern_type.940805.4
// CHECK:STDOUT:   %return.param_patt.loc13_47.2 => constants.%return.param_patt.bcf038.2
// CHECK:STDOUT:   %return.patt.loc13_34.2 => constants.%return.patt.690db9.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EnclosingTemplate(constants.%S) {
// CHECK:STDOUT:   %T.patt.loc8_32.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_32.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a55
// CHECK:STDOUT:   %a.patt.loc9_8.2 => constants.%a.patt.2a2
// CHECK:STDOUT:   %.loc9_15.3 => constants.%inst.name_ref.426619.1
// CHECK:STDOUT:   %.loc9_15.4 => constants.%S.Make.type
// CHECK:STDOUT:   %.loc9_15.5 => constants.%S.Make
// CHECK:STDOUT:   %.loc9_21.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc9_21.5 => constants.%S
// CHECK:STDOUT:   %.loc9_21.6 => <not constant>
// CHECK:STDOUT:   %.loc9_21.7 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_21.8 => <not constant>
// CHECK:STDOUT:   %Local.type => constants.%Local.type.2cd
// CHECK:STDOUT:   %Local => constants.%Local.6cd
// CHECK:STDOUT:   %.loc17_17.3 => constants.%inst.name_ref.426619.2
// CHECK:STDOUT:   %.loc17_17.4 => constants.%S.Make.type
// CHECK:STDOUT:   %.loc17_17.5 => constants.%S.Make
// CHECK:STDOUT:   %.loc17_23.3 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc17_23.4 => constants.%S
// CHECK:STDOUT:   %.loc17_23.5 => <not constant>
// CHECK:STDOUT:   %.loc17_27.3 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc17_27.4 => constants.%i32
// CHECK:STDOUT:   %.loc17_27.5 => <not constant>
// CHECK:STDOUT:   %.loc17_29.3 => constants.%inst.splice_block.426619.7
// CHECK:STDOUT:   %.loc17_29.4 => constants.%i32
// CHECK:STDOUT:   %.loc17_29.5 => <not constant>
// CHECK:STDOUT:   %.loc17_30.2 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc17_30.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local(constants.%S) {
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %pattern_type.loc13_13 => constants.%pattern_type.a55
// CHECK:STDOUT:   %_.param_patt.loc13_13.2 => constants.%_.param_patt.230
// CHECK:STDOUT:   %_.patt.loc13_13.2 => constants.%_.patt.e4c
// CHECK:STDOUT:   %.loc13_29.1 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc13_29.2 => constants.%i32
// CHECK:STDOUT:   %.loc13_29.3 => <not constant>
// CHECK:STDOUT:   %pattern_type.loc13_19 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %y.param_patt.loc13_19.2 => constants.%y.param_patt.673
// CHECK:STDOUT:   %y.patt.loc13_19.2 => constants.%y.patt.547
// CHECK:STDOUT:   %.loc13_45.1 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc13_45.2 => constants.%i32
// CHECK:STDOUT:   %.loc13_45.3 => <not constant>
// CHECK:STDOUT:   %.loc13_47.1 => constants.%.795f
// CHECK:STDOUT:   %pattern_type.loc13_47 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc13_47.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc13_34.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_13 => constants.%complete_type.cdf
// CHECK:STDOUT:   %require_complete.loc13_19 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc13_34 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc14_13.2 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.9
// CHECK:STDOUT:   %.loc14_13.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
