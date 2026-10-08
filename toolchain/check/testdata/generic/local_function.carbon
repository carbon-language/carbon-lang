// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/local_function.carbon

fn Test[unused T: type]() {
  fn Local() {
  }
  Local();
}

class S {
  fn Make() -> S { return {.n = 42}; }
  var n: i32;
}

fn EnclosingTemplate(template T: type) -> i32 {
  let a: T = T.Make();
  fn Local(_: T, y: typeof(a.n)) -> typeof(a.n) {
    return y;
  }
  return Local(T.Make(), a.n);
}

fn CallEnclosingTemplate() -> i32 {
  return EnclosingTemplate(S);
}

// CHECK:STDOUT: --- local_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %Local.type.c16: type = fn_type @Local.loc16, @Test(%T.67db0b.1) [symbolic]
// CHECK:STDOUT:   %Local.76d: %Local.type.c16 = struct_value () [symbolic]
// CHECK:STDOUT:   %Local.specific_fn.145: <specific function> = specific_function %Local.76d, @Local.loc16(%T.67db0b.1) [symbolic]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %.e6c: Core.Form = init_form %S [concrete]
// CHECK:STDOUT:   %pattern_type.a55: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %return.param_patt.098: %pattern_type.a55 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.289: %pattern_type.a55 = return_slot_pattern %return.param_patt.098, %S [concrete]
// CHECK:STDOUT:   %S.Make.type: type = fn_type @S.Make [concrete]
// CHECK:STDOUT:   %S.Make: %S.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %S.elem: type = unbound_element_type %S, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.9ed: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n.9ed [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %struct_type.n.44a: type = struct_type {.n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.44a = struct_value (%int_42.20e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.098: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %S.val: %S = struct_value (%int_42.ac4) [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.2: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %EnclosingTemplate.type: type = fn_type @EnclosingTemplate [concrete]
// CHECK:STDOUT:   %EnclosingTemplate: %EnclosingTemplate.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.2 [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %T.67db0b.2 [template]
// CHECK:STDOUT:   %a.patt.54f: %pattern_type.51d1c4.2 = value_binding_pattern a [template]
// CHECK:STDOUT:   %.6951bd.1: type = type_of_inst @EnclosingTemplate.%.loc27_15.3 [template]
// CHECK:STDOUT:   %.e6bb03.1: %.6951bd.1 = splice_inst @EnclosingTemplate.%.loc27_15.3 [template]
// CHECK:STDOUT:   %.5e1a30.1: type = type_of_inst @EnclosingTemplate.%.loc27_21.4 [template]
// CHECK:STDOUT:   %.03fafa.1: %.5e1a30.1 = splice_inst @EnclosingTemplate.%.loc27_21.4 [template]
// CHECK:STDOUT:   %.ce5: %T.67db0b.2 = splice_inst @EnclosingTemplate.%.loc27_21.7 [template]
// CHECK:STDOUT:   %_.param_patt.339: %pattern_type.51d1c4.2 = value_param_pattern [template]
// CHECK:STDOUT:   %_.patt.ace: %pattern_type.51d1c4.2 = wrapper_binding_pattern _, %_.param_patt.339 [template]
// CHECK:STDOUT:   %.237907.1: type = type_of_inst @Local.loc28.%.loc28_29.1 [template]
// CHECK:STDOUT:   %.f9658d.1: %.237907.1 = splice_inst @Local.loc28.%.loc28_29.1 [template]
// CHECK:STDOUT:   %pattern_type.940805.1: type = pattern_type %.237907.1 [template]
// CHECK:STDOUT:   %y.param_patt.8246e7.1: %pattern_type.940805.1 = value_param_pattern [template]
// CHECK:STDOUT:   %y.patt.d967f4.1: %pattern_type.940805.1 = wrapper_binding_pattern y, %y.param_patt.8246e7.1 [template]
// CHECK:STDOUT:   %.237907.2: type = type_of_inst @Local.loc28.%.loc28_45.1 [template]
// CHECK:STDOUT:   %.f9658d.2: %.237907.2 = splice_inst @Local.loc28.%.loc28_45.1 [template]
// CHECK:STDOUT:   %.6c4209.1: Core.Form = init_form %.237907.2 [template]
// CHECK:STDOUT:   %pattern_type.940805.2: type = pattern_type %.237907.2 [template]
// CHECK:STDOUT:   %return.param_patt.bcf038.1: %pattern_type.940805.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.690db9.1: %pattern_type.940805.2 = return_slot_pattern %return.param_patt.bcf038.1, %.237907.2 [template]
// CHECK:STDOUT:   %.e9daaa.1: <instruction> = access_member_action @Local.loc28.%a.ref.loc28_28, n [template]
// CHECK:STDOUT:   %.237907.3: type = type_of_inst %.e9daaa.1 [template]
// CHECK:STDOUT:   %.f9658d.3: %.237907.3 = splice_inst %.e9daaa.1 [template]
// CHECK:STDOUT:   %pattern_type.940805.3: type = pattern_type %.237907.3 [template]
// CHECK:STDOUT:   %y.param_patt.8246e7.2: %pattern_type.940805.3 = value_param_pattern [template]
// CHECK:STDOUT:   %y.patt.d967f4.2: %pattern_type.940805.3 = wrapper_binding_pattern y, %y.param_patt.8246e7.2 [template]
// CHECK:STDOUT:   %.e9daaa.2: <instruction> = access_member_action @Local.loc28.%a.ref.loc28_44, n [template]
// CHECK:STDOUT:   %.237907.4: type = type_of_inst %.e9daaa.2 [template]
// CHECK:STDOUT:   %.f9658d.4: %.237907.4 = splice_inst %.e9daaa.2 [template]
// CHECK:STDOUT:   %.6c4209.2: Core.Form = init_form %.237907.4 [template]
// CHECK:STDOUT:   %pattern_type.940805.4: type = pattern_type %.237907.4 [template]
// CHECK:STDOUT:   %return.param_patt.bcf038.2: %pattern_type.940805.4 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.690db9.2: %pattern_type.940805.4 = return_slot_pattern %return.param_patt.bcf038.2, %.237907.4 [template]
// CHECK:STDOUT:   %Local.type.348: type = fn_type @Local.loc28, @EnclosingTemplate(%T.67db0b.2) [template]
// CHECK:STDOUT:   %Local.cb2: %Local.type.348 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.b43ae1.1: <witness> = require_complete_type %.237907.1 [template]
// CHECK:STDOUT:   %require_complete.b43ae1.2: <witness> = require_complete_type %.237907.2 [template]
// CHECK:STDOUT:   %tuple.type.f7a: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.26b: %.237907.2 = splice_inst @Local.loc28.%tuple.elem0 [template]
// CHECK:STDOUT:   %.6951bd.2: type = type_of_inst @EnclosingTemplate.%.loc31_17.3 [template]
// CHECK:STDOUT:   %.e6bb03.2: %.6951bd.2 = splice_inst @EnclosingTemplate.%.loc31_17.3 [template]
// CHECK:STDOUT:   %.5e1a30.2: type = type_of_inst @EnclosingTemplate.%.loc31_23.3 [template]
// CHECK:STDOUT:   %.03fafa.2: %.5e1a30.2 = splice_inst @EnclosingTemplate.%.loc31_23.3 [template]
// CHECK:STDOUT:   %.237907.5: type = type_of_inst @EnclosingTemplate.%.loc31_27.3 [template]
// CHECK:STDOUT:   %.f9658d.5: %.237907.5 = splice_inst @EnclosingTemplate.%.loc31_27.3 [template]
// CHECK:STDOUT:   %.897: type = type_of_inst @EnclosingTemplate.%.loc31_29.3 [template]
// CHECK:STDOUT:   %.c25: %.897 = splice_inst @EnclosingTemplate.%.loc31_29.3 [template]
// CHECK:STDOUT:   %.28a: %i32 = splice_inst @EnclosingTemplate.%.loc31_30.2 [template]
// CHECK:STDOUT:   %CallEnclosingTemplate.type: type = fn_type @CallEnclosingTemplate [concrete]
// CHECK:STDOUT:   %CallEnclosingTemplate: %CallEnclosingTemplate.type = struct_value () [concrete]
// CHECK:STDOUT:   %EnclosingTemplate.specific_fn: <specific function> = specific_function %EnclosingTemplate, @EnclosingTemplate(%S) [concrete]
// CHECK:STDOUT:   %a.patt.2a2: %pattern_type.a55 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %inst.name_ref.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Make.ref.b85793.1: %S.Make.type = name_ref Make, @S.%S.Make.decl [concrete = %S.Make]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.058268.1: init %S = splice_block %S.Make.call.2c2051.2 {
// CHECK:STDOUT:       %.4056a4.2: ref %S = temporary_storage
// CHECK:STDOUT:       %S.Make.call.2c2051.2: init %S to %.4056a4.2 = call %S.Make()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.349: %S = splice_block %.ff4281.2 {
// CHECK:STDOUT:       %.995c26.2: ref %S = temporary %.4056a4.2, %.058268.1
// CHECK:STDOUT:       %.ff4281.2: %S = acquire_value %.995c26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Local.type.2cd: type = fn_type @Local.loc28, @EnclosingTemplate(%S) [concrete]
// CHECK:STDOUT:   %Local.6cd: %Local.type.2cd = struct_value () [concrete]
// CHECK:STDOUT:   %inst.name_ref.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Make.ref.b85793.2: %S.Make.type = name_ref Make, @S.%S.Make.decl [concrete = %S.Make]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.058268.2: init %S = splice_block %S.Make.call.2c2051.1 {
// CHECK:STDOUT:       %.4056a4.1: ref %S = temporary_storage
// CHECK:STDOUT:       %S.Make.call.2c2051.1: init %S to %.4056a4.1 = call %S.Make()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7eb: %i32 = splice_block %.5ea {
// CHECK:STDOUT:       %.ddc: %S = specific_inst @EnclosingTemplate.%a.ref, @EnclosingTemplate(%S)
// CHECK:STDOUT:       %n.ref.1bbe5e.3: %S.elem = name_ref n, @S.%field_decl [concrete = @S.%field_decl]
// CHECK:STDOUT:       %.d45: ref %i32 = class_element_access %.ddc, element0
// CHECK:STDOUT:       %.5ea: %i32 = acquire_value %.d45
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.param_patt.230: %pattern_type.a55 = value_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.e4c: %pattern_type.a55 = wrapper_binding_pattern _, %_.param_patt.230 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.591303.1: %i32 = splice_block %.dcb12d.2 {
// CHECK:STDOUT:       %.ac15a1.2: %S = specific_inst @Local.loc28.%a.ref.loc28_28, @Local.loc28(%S)
// CHECK:STDOUT:       %n.ref.1bbe5e.2: %S.elem = name_ref n, @S.%field_decl [concrete = @S.%field_decl]
// CHECK:STDOUT:       %.817716.2: ref %i32 = class_element_access %.ac15a1.2, element0
// CHECK:STDOUT:       %.dcb12d.2: %i32 = acquire_value %.817716.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.param_patt.673: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.547: %pattern_type.6b6 = wrapper_binding_pattern y, %y.param_patt.673 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.591303.2: %i32 = splice_block %.dcb12d.1 {
// CHECK:STDOUT:       %.ac15a1.1: %S = specific_inst @Local.loc28.%a.ref.loc28_44, @Local.loc28(%S)
// CHECK:STDOUT:       %n.ref.1bbe5e.1: %S.elem = name_ref n, @S.%field_decl [concrete = @S.%field_decl]
// CHECK:STDOUT:       %.817716.1: ref %i32 = class_element_access %.ac15a1.1, element0
// CHECK:STDOUT:       %.dcb12d.1: %i32 = acquire_value %.817716.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Local.specific_fn.da9c17.1: <specific function> = specific_function %Local.6cd, @Local.loc28(%S) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.10e: init %i32 = splice_block %Local.call {
// CHECK:STDOUT:       %Local.specific_fn.da9c17.2: <specific function> = specific_function %Local.6cd, @Local.loc28(%S) [concrete = %Local.specific_fn.da9c17.1]
// CHECK:STDOUT:       %.995c26.1: ref %S = temporary %.4056a4.1, %.058268.2
// CHECK:STDOUT:       %.ff4281.1: %S = acquire_value %.995c26.1
// CHECK:STDOUT:       %Local.call: init %i32 = call %Local.specific_fn.da9c17.2(%.ff4281.1, %.7eb)
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
// CHECK:STDOUT:       %.185: %i32 = specific_inst @Local.loc28.%y.ref, @Local.loc28(%S)
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
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     .EnclosingTemplate = %EnclosingTemplate.decl
// CHECK:STDOUT:     .CallEnclosingTemplate = %CallEnclosingTemplate.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %T.patt.loc15_17.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_17.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_19.1: type = splice_block %.loc15_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_17.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %EnclosingTemplate.decl: %EnclosingTemplate.type = fn_decl @EnclosingTemplate [concrete = constants.%EnclosingTemplate] {
// CHECK:STDOUT:     %T.patt.loc26_32.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc26_32.2 (constants.%T.patt.3a0592.2)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc26_43: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %.loc26_34.1: type = splice_block %.loc26_34.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc26_34.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_32.2: type = symbolic_binding T, 0, template [template = %T.loc26_32.1 (constants.%T.67db0b.2)]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallEnclosingTemplate.decl: %CallEnclosingTemplate.type = fn_decl @CallEnclosingTemplate [concrete = constants.%CallEnclosingTemplate] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc34: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @S {
// CHECK:STDOUT:   %S.Make.decl: %S.Make.type = fn_decl @S.Make [concrete = constants.%S.Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a55 = out_param_pattern [concrete = constants.%return.param_patt.098]
// CHECK:STDOUT:     %return.patt: %pattern_type.a55 = return_slot_pattern %return.param_patt, %S.ref [concrete = constants.%return.patt.289]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:     %.loc22_16: Core.Form = init_form %S.ref [concrete = constants.%.e6c]
// CHECK:STDOUT:     %return.param: ref %S = out_param call_param0
// CHECK:STDOUT:     %return: ref %S = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl: %S.elem = field_decl n, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.9ed [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%S
// CHECK:STDOUT:   .S = <poisoned>
// CHECK:STDOUT:   .Make = %S.Make.decl
// CHECK:STDOUT:   .n = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc15_17.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_17.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_17.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc15_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_17.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Local.type: type = fn_type @Local.loc16, @Test(%T.loc15_17.1) [symbolic = %Local.type (constants.%Local.type.c16)]
// CHECK:STDOUT:   %Local: @Test.%Local.type (%Local.type.c16) = struct_value () [symbolic = %Local (constants.%Local.76d)]
// CHECK:STDOUT:   %Local.specific_fn.loc18_3.2: <specific function> = specific_function %Local, @Local.loc16(%T.loc15_17.1) [symbolic = %Local.specific_fn.loc18_3.2 (constants.%Local.specific_fn.145)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Local.decl: @Test.%Local.type (%Local.type.c16) = fn_decl @Local.loc16 [symbolic = @Test.%Local (constants.%Local.76d)] {} {}
// CHECK:STDOUT:     %Local.ref: @Test.%Local.type (%Local.type.c16) = name_ref Local, %Local.decl [symbolic = %Local (constants.%Local.76d)]
// CHECK:STDOUT:     %Local.specific_fn.loc18_3.1: <specific function> = specific_function %Local.ref, @Local.loc16(constants.%T.67db0b.1) [symbolic = %Local.specific_fn.loc18_3.2 (constants.%Local.specific_fn.145)]
// CHECK:STDOUT:     %Local.call: init %empty_tuple.type = call %Local.specific_fn.loc18_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Local.loc16(@Test.%T.loc15_17.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @S.Make() -> out %return.param: %S {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %.loc22_35.1: %struct_type.n.44a = struct_literal (%int_42) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc22_35.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_35.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method.098]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc22_35.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc22_35.2: init %i32 = converted %int_42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc22_35.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc22_35.4: init %i32 to %.loc22_35.3 = in_place_init %.loc22_35.2 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc22_35.5: init %S to %return.param = class_init (%.loc22_35.4) [concrete = constants.%S.val]
// CHECK:STDOUT:   %.loc22_36: init %S = converted %.loc22_35.1, %.loc22_35.5 [concrete = constants.%S.val]
// CHECK:STDOUT:   return %.loc22_36 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EnclosingTemplate(%T.loc26_32.2: type) {
// CHECK:STDOUT:   %T.patt.loc26_32.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc26_32.2 (constants.%T.patt.3a0592.2)]
// CHECK:STDOUT:   %T.loc26_32.1: type = symbolic_binding T, 0, template [template = %T.loc26_32.1 (constants.%T.67db0b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc26_32.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc26_32.1 [template = %pattern_type (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:   %a.patt.loc27_8.2: @EnclosingTemplate.%pattern_type (%pattern_type.51d1c4.2) = value_binding_pattern a [template = %a.patt.loc27_8.2 (constants.%a.patt.54f)]
// CHECK:STDOUT:   %.loc27_15.3: <instruction> = access_member_action %T.ref.loc27_14, Make [template]
// CHECK:STDOUT:   %.loc27_15.4: type = type_of_inst %.loc27_15.3 [template = %.loc27_15.4 (constants.%.6951bd.1)]
// CHECK:STDOUT:   %.loc27_15.5: @EnclosingTemplate.%.loc27_15.4 (%.6951bd.1) = splice_inst %.loc27_15.3 [template = %.loc27_15.5 (constants.%.e6bb03.1)]
// CHECK:STDOUT:   %.loc27_21.4: <instruction> = call_action (%.loc27_15.2), false [template]
// CHECK:STDOUT:   %.loc27_21.5: type = type_of_inst %.loc27_21.4 [template = %.loc27_21.5 (constants.%.5e1a30.1)]
// CHECK:STDOUT:   %.loc27_21.6: @EnclosingTemplate.%.loc27_21.5 (%.5e1a30.1) = splice_inst %.loc27_21.4 [template = %.loc27_21.6 (constants.%.03fafa.1)]
// CHECK:STDOUT:   %.loc27_21.7: <instruction> = convert_to_value_action %.loc27_21.2, %T.loc26_32.1 [template]
// CHECK:STDOUT:   %.loc27_21.8: @EnclosingTemplate.%T.loc26_32.1 (%T.67db0b.2) = splice_inst %.loc27_21.7 [template = %.loc27_21.8 (constants.%.ce5)]
// CHECK:STDOUT:   %Local.type: type = fn_type @Local.loc28, @EnclosingTemplate(%T.loc26_32.1) [template = %Local.type (constants.%Local.type.348)]
// CHECK:STDOUT:   %Local: @EnclosingTemplate.%Local.type (%Local.type.348) = struct_value () [template = %Local (constants.%Local.cb2)]
// CHECK:STDOUT:   %.loc31_17.3: <instruction> = access_member_action %T.ref.loc31, Make [template]
// CHECK:STDOUT:   %.loc31_17.4: type = type_of_inst %.loc31_17.3 [template = %.loc31_17.4 (constants.%.6951bd.2)]
// CHECK:STDOUT:   %.loc31_17.5: @EnclosingTemplate.%.loc31_17.4 (%.6951bd.2) = splice_inst %.loc31_17.3 [template = %.loc31_17.5 (constants.%.e6bb03.2)]
// CHECK:STDOUT:   %.loc31_23.3: <instruction> = call_action (%.loc31_17.2), false [template]
// CHECK:STDOUT:   %.loc31_23.4: type = type_of_inst %.loc31_23.3 [template = %.loc31_23.4 (constants.%.5e1a30.2)]
// CHECK:STDOUT:   %.loc31_23.5: @EnclosingTemplate.%.loc31_23.4 (%.5e1a30.2) = splice_inst %.loc31_23.3 [template = %.loc31_23.5 (constants.%.03fafa.2)]
// CHECK:STDOUT:   %.loc31_27.3: <instruction> = access_member_action %a.ref, n [template]
// CHECK:STDOUT:   %.loc31_27.4: type = type_of_inst %.loc31_27.3 [template = %.loc31_27.4 (constants.%.237907.5)]
// CHECK:STDOUT:   %.loc31_27.5: @EnclosingTemplate.%.loc31_27.4 (%.237907.5) = splice_inst %.loc31_27.3 [template = %.loc31_27.5 (constants.%.f9658d.5)]
// CHECK:STDOUT:   %.loc31_29.3: <instruction> = call_action (%Local.ref, %.loc31_23.2, %.loc31_27.2), false [template]
// CHECK:STDOUT:   %.loc31_29.4: type = type_of_inst %.loc31_29.3 [template = %.loc31_29.4 (constants.%.897)]
// CHECK:STDOUT:   %.loc31_29.5: @EnclosingTemplate.%.loc31_29.4 (%.897) = splice_inst %.loc31_29.3 [template = %.loc31_29.5 (constants.%.c25)]
// CHECK:STDOUT:   %.loc31_30.2: <instruction> = convert_action %.loc31_29.2, constants.%i32, element10 [template]
// CHECK:STDOUT:   %.loc31_30.3: %i32 = splice_inst %.loc31_30.2 [template = %.loc31_30.3 (constants.%.28a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc27_14: type = name_ref T, %T.loc26_32.2 [template = %T.loc26_32.1 (constants.%T.67db0b.2)]
// CHECK:STDOUT:     %.loc27_15.1: type = type_of_inst %.loc27_15.3 [template = %.loc27_15.4 (constants.%.6951bd.1)]
// CHECK:STDOUT:     %.loc27_15.2: @EnclosingTemplate.%.loc27_15.4 (%.6951bd.1) = splice_inst %.loc27_15.3 [template = %.loc27_15.5 (constants.%.e6bb03.1)]
// CHECK:STDOUT:     %.loc27_21.1: type = type_of_inst %.loc27_21.4 [template = %.loc27_21.5 (constants.%.5e1a30.1)]
// CHECK:STDOUT:     %.loc27_21.2: @EnclosingTemplate.%.loc27_21.5 (%.5e1a30.1) = splice_inst %.loc27_21.4 [template = %.loc27_21.6 (constants.%.03fafa.1)]
// CHECK:STDOUT:     %.loc27_21.3: @EnclosingTemplate.%T.loc26_32.1 (%T.67db0b.2) = splice_inst %.loc27_21.7 [template = %.loc27_21.8 (constants.%.ce5)]
// CHECK:STDOUT:     %T.ref.loc27_10: type = name_ref T, %T.loc26_32.2 [template = %T.loc26_32.1 (constants.%T.67db0b.2)]
// CHECK:STDOUT:     %a: @EnclosingTemplate.%T.loc26_32.1 (%T.67db0b.2) = wrapper_binding a, %.loc27_21.3
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt.loc27_8.1: @EnclosingTemplate.%pattern_type (%pattern_type.51d1c4.2) = value_binding_pattern a [template = %a.patt.loc27_8.2 (constants.%a.patt.54f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Local.decl: @EnclosingTemplate.%Local.type (%Local.type.348) = fn_decl @Local.loc28 [template = @EnclosingTemplate.%Local (constants.%Local.cb2)] {
// CHECK:STDOUT:       %_.param_patt.loc28_13.1: @Local.loc28.%pattern_type.loc28_13 (%pattern_type.51d1c4.2) = value_param_pattern [template = %_.param_patt.loc28_13.2 (constants.%_.param_patt.339)]
// CHECK:STDOUT:       %_.patt.loc28_13.1: @Local.loc28.%pattern_type.loc28_13 (%pattern_type.51d1c4.2) = wrapper_binding_pattern _, %_.param_patt.loc28_13.1 [template = %_.patt.loc28_13.2 (constants.%_.patt.ace)]
// CHECK:STDOUT:       %y.param_patt.loc28_19.1: @Local.loc28.%pattern_type.loc28_19 (%pattern_type.940805.1) = value_param_pattern [template = %y.param_patt.loc28_19.2 (constants.%y.param_patt.8246e7.1)]
// CHECK:STDOUT:       %y.patt.loc28_19.1: @Local.loc28.%pattern_type.loc28_19 (%pattern_type.940805.1) = wrapper_binding_pattern y, %y.param_patt.loc28_19.1 [template = %y.patt.loc28_19.2 (constants.%y.patt.d967f4.1)]
// CHECK:STDOUT:       %return.param_patt.loc28_47.1: @Local.loc28.%pattern_type.loc28_47 (%pattern_type.940805.2) = out_param_pattern [template = %return.param_patt.loc28_47.2 (constants.%return.param_patt.bcf038.1)]
// CHECK:STDOUT:       %return.patt.loc28_34.1: @Local.loc28.%pattern_type.loc28_47 (%pattern_type.940805.2) = return_slot_pattern %return.param_patt.loc28_47.1, %.loc28_47.2 [template = %return.patt.loc28_34.2 (constants.%return.patt.690db9.1)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc28_47.2: type = type_of %.loc28_45.5 in [template = %.loc28_45.2 (constants.%.237907.2)] {
// CHECK:STDOUT:         %.loc28_44: @Local.loc28.%T (%T.67db0b.2) = specific_inst @EnclosingTemplate.%a, @EnclosingTemplate(constants.%T.67db0b.2)
// CHECK:STDOUT:         %a.ref.loc28_44: @Local.loc28.%T (%T.67db0b.2) = name_ref a, %.loc28_44
// CHECK:STDOUT:         %.loc28_45.4: type = type_of_inst %.loc28_45.1 [template = %.loc28_45.2 (constants.%.237907.2)]
// CHECK:STDOUT:         %.loc28_45.5: @Local.loc28.%.loc28_45.2 (%.237907.2) = splice_inst %.loc28_45.1 [template = %.loc28_45.3 (constants.%.f9658d.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.loc28_47.3: Core.Form = init_form %.loc28_47.2 [template = %.loc28_47.1 (constants.%.6c4209.1)]
// CHECK:STDOUT:       %_.param: @Local.loc28.%T (%T.67db0b.2) = value_param call_param0
// CHECK:STDOUT:       %.loc28_15.1: type = splice_block %T.ref [template = %T (constants.%T.67db0b.2)] {
// CHECK:STDOUT:         %.loc28_15.2: type = specific_constant @EnclosingTemplate.%T.loc26_32.2, @EnclosingTemplate(constants.%T.67db0b.2) [template = %T (constants.%T.67db0b.2)]
// CHECK:STDOUT:         %T.ref: type = name_ref T, %.loc28_15.2 [template = %T (constants.%T.67db0b.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %_: @Local.loc28.%T (%T.67db0b.2) = wrapper_binding _, %_.param
// CHECK:STDOUT:       %y.param: @Local.loc28.%.loc28_29.2 (%.237907.1) = value_param call_param1
// CHECK:STDOUT:       %.loc28_31: type = type_of %.loc28_29.5 in [template = %.loc28_29.2 (constants.%.237907.1)] {
// CHECK:STDOUT:         %.loc28_28: @Local.loc28.%T (%T.67db0b.2) = specific_inst @EnclosingTemplate.%a, @EnclosingTemplate(constants.%T.67db0b.2)
// CHECK:STDOUT:         %a.ref.loc28_28: @Local.loc28.%T (%T.67db0b.2) = name_ref a, %.loc28_28
// CHECK:STDOUT:         %.loc28_29.4: type = type_of_inst %.loc28_29.1 [template = %.loc28_29.2 (constants.%.237907.1)]
// CHECK:STDOUT:         %.loc28_29.5: @Local.loc28.%.loc28_29.2 (%.237907.1) = splice_inst %.loc28_29.1 [template = %.loc28_29.3 (constants.%.f9658d.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @Local.loc28.%.loc28_29.2 (%.237907.1) = wrapper_binding y, %y.param
// CHECK:STDOUT:       %return.param: ref @Local.loc28.%.loc28_45.2 (%.237907.2) = out_param call_param2
// CHECK:STDOUT:       %return: ref @Local.loc28.%.loc28_45.2 (%.237907.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Local.ref: @EnclosingTemplate.%Local.type (%Local.type.348) = name_ref Local, %Local.decl [template = %Local (constants.%Local.cb2)]
// CHECK:STDOUT:     %T.ref.loc31: type = name_ref T, %T.loc26_32.2 [template = %T.loc26_32.1 (constants.%T.67db0b.2)]
// CHECK:STDOUT:     %.loc31_17.1: type = type_of_inst %.loc31_17.3 [template = %.loc31_17.4 (constants.%.6951bd.2)]
// CHECK:STDOUT:     %.loc31_17.2: @EnclosingTemplate.%.loc31_17.4 (%.6951bd.2) = splice_inst %.loc31_17.3 [template = %.loc31_17.5 (constants.%.e6bb03.2)]
// CHECK:STDOUT:     %.loc31_23.1: type = type_of_inst %.loc31_23.3 [template = %.loc31_23.4 (constants.%.5e1a30.2)]
// CHECK:STDOUT:     %.loc31_23.2: @EnclosingTemplate.%.loc31_23.4 (%.5e1a30.2) = splice_inst %.loc31_23.3 [template = %.loc31_23.5 (constants.%.03fafa.2)]
// CHECK:STDOUT:     %a.ref: @EnclosingTemplate.%T.loc26_32.1 (%T.67db0b.2) = name_ref a, %a
// CHECK:STDOUT:     %.loc31_27.1: type = type_of_inst %.loc31_27.3 [template = %.loc31_27.4 (constants.%.237907.5)]
// CHECK:STDOUT:     %.loc31_27.2: @EnclosingTemplate.%.loc31_27.4 (%.237907.5) = splice_inst %.loc31_27.3 [template = %.loc31_27.5 (constants.%.f9658d.5)]
// CHECK:STDOUT:     %.loc31_29.1: type = type_of_inst %.loc31_29.3 [template = %.loc31_29.4 (constants.%.897)]
// CHECK:STDOUT:     %.loc31_29.2: @EnclosingTemplate.%.loc31_29.4 (%.897) = splice_inst %.loc31_29.3 [template = %.loc31_29.5 (constants.%.c25)]
// CHECK:STDOUT:     %.loc31_30.1: %i32 = splice_inst %.loc31_30.2 [template = %.loc31_30.3 (constants.%.28a)]
// CHECK:STDOUT:     return %.loc31_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Local.loc28(@EnclosingTemplate.%T.loc26_32.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T.67db0b.2)]
// CHECK:STDOUT:   %pattern_type.loc28_13: type = pattern_type %T [template = %pattern_type.loc28_13 (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:   %_.param_patt.loc28_13.2: @Local.loc28.%pattern_type.loc28_13 (%pattern_type.51d1c4.2) = value_param_pattern [template = %_.param_patt.loc28_13.2 (constants.%_.param_patt.339)]
// CHECK:STDOUT:   %_.patt.loc28_13.2: @Local.loc28.%pattern_type.loc28_13 (%pattern_type.51d1c4.2) = wrapper_binding_pattern _, %_.param_patt.loc28_13.2 [template = %_.patt.loc28_13.2 (constants.%_.patt.ace)]
// CHECK:STDOUT:   %.loc28_29.1: <instruction> = access_member_action %a.ref.loc28_28, n [template]
// CHECK:STDOUT:   %.loc28_29.2: type = type_of_inst %.loc28_29.1 [template = %.loc28_29.2 (constants.%.237907.1)]
// CHECK:STDOUT:   %.loc28_29.3: @Local.loc28.%.loc28_29.2 (%.237907.1) = splice_inst %.loc28_29.1 [template = %.loc28_29.3 (constants.%.f9658d.1)]
// CHECK:STDOUT:   %pattern_type.loc28_19: type = pattern_type %.loc28_29.2 [template = %pattern_type.loc28_19 (constants.%pattern_type.940805.1)]
// CHECK:STDOUT:   %y.param_patt.loc28_19.2: @Local.loc28.%pattern_type.loc28_19 (%pattern_type.940805.1) = value_param_pattern [template = %y.param_patt.loc28_19.2 (constants.%y.param_patt.8246e7.1)]
// CHECK:STDOUT:   %y.patt.loc28_19.2: @Local.loc28.%pattern_type.loc28_19 (%pattern_type.940805.1) = wrapper_binding_pattern y, %y.param_patt.loc28_19.2 [template = %y.patt.loc28_19.2 (constants.%y.patt.d967f4.1)]
// CHECK:STDOUT:   %.loc28_45.1: <instruction> = access_member_action %a.ref.loc28_44, n [template]
// CHECK:STDOUT:   %.loc28_45.2: type = type_of_inst %.loc28_45.1 [template = %.loc28_45.2 (constants.%.237907.2)]
// CHECK:STDOUT:   %.loc28_45.3: @Local.loc28.%.loc28_45.2 (%.237907.2) = splice_inst %.loc28_45.1 [template = %.loc28_45.3 (constants.%.f9658d.2)]
// CHECK:STDOUT:   %.loc28_47.1: Core.Form = init_form %.loc28_45.2 [template = %.loc28_47.1 (constants.%.6c4209.1)]
// CHECK:STDOUT:   %pattern_type.loc28_47: type = pattern_type %.loc28_45.2 [template = %pattern_type.loc28_47 (constants.%pattern_type.940805.2)]
// CHECK:STDOUT:   %return.param_patt.loc28_47.2: @Local.loc28.%pattern_type.loc28_47 (%pattern_type.940805.2) = out_param_pattern [template = %return.param_patt.loc28_47.2 (constants.%return.param_patt.bcf038.1)]
// CHECK:STDOUT:   %return.patt.loc28_34.2: @Local.loc28.%pattern_type.loc28_47 (%pattern_type.940805.2) = return_slot_pattern %return.param_patt.loc28_47.2, %.loc28_45.2 [template = %return.patt.loc28_34.2 (constants.%return.patt.690db9.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_13: <witness> = require_complete_type %T [template = %require_complete.loc28_13 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc28_19: <witness> = require_complete_type %.loc28_29.2 [template = %require_complete.loc28_19 (constants.%require_complete.b43ae1.1)]
// CHECK:STDOUT:   %require_complete.loc28_34: <witness> = require_complete_type %.loc28_45.2 [template = %require_complete.loc28_34 (constants.%require_complete.b43ae1.2)]
// CHECK:STDOUT:   %.loc29_13.2: %tuple.type.f7a = initialize_action %y.ref, %.loc28_45.2, %return.param, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc29_13.2, element0 [template]
// CHECK:STDOUT:   %.loc29_13.3: @Local.loc28.%.loc28_45.2 (%.237907.2) = splice_inst %tuple.elem0 [template = %.loc29_13.3 (constants.%.26b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: @Local.loc28.%T (%T.67db0b.2), %y.param: @Local.loc28.%.loc28_29.2 (%.237907.1)) -> out %return.param: @Local.loc28.%.loc28_45.2 (%.237907.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %y.ref: @Local.loc28.%.loc28_29.2 (%.237907.1) = name_ref y, %y
// CHECK:STDOUT:     %.loc29_13.1: @Local.loc28.%.loc28_45.2 (%.237907.2) = splice_inst %tuple.elem0 [template = %.loc29_13.3 (constants.%.26b)]
// CHECK:STDOUT:     return %.loc29_13.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallEnclosingTemplate() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EnclosingTemplate.ref: %EnclosingTemplate.type = name_ref EnclosingTemplate, file.%EnclosingTemplate.decl [concrete = constants.%EnclosingTemplate]
// CHECK:STDOUT:   %S.ref: type = name_ref S, file.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %EnclosingTemplate.specific_fn: <specific function> = specific_function %EnclosingTemplate.ref, @EnclosingTemplate(constants.%S) [concrete = constants.%EnclosingTemplate.specific_fn]
// CHECK:STDOUT:   %EnclosingTemplate.call: init %i32 = call %EnclosingTemplate.specific_fn()
// CHECK:STDOUT:   return %EnclosingTemplate.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc15_17.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc15_17.1 => constants.%T.67db0b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local.loc16(constants.%T.67db0b.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EnclosingTemplate(constants.%T.67db0b.2) {
// CHECK:STDOUT:   %T.patt.loc26_32.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc26_32.1 => constants.%T.67db0b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local.loc28(constants.%T.67db0b.2) {
// CHECK:STDOUT:   %T => constants.%T.67db0b.2
// CHECK:STDOUT:   %pattern_type.loc28_13 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %_.param_patt.loc28_13.2 => constants.%_.param_patt.339
// CHECK:STDOUT:   %_.patt.loc28_13.2 => constants.%_.patt.ace
// CHECK:STDOUT:   %.loc28_29.1 => constants.%.e9daaa.1
// CHECK:STDOUT:   %.loc28_29.2 => constants.%.237907.3
// CHECK:STDOUT:   %.loc28_29.3 => constants.%.f9658d.3
// CHECK:STDOUT:   %pattern_type.loc28_19 => constants.%pattern_type.940805.3
// CHECK:STDOUT:   %y.param_patt.loc28_19.2 => constants.%y.param_patt.8246e7.2
// CHECK:STDOUT:   %y.patt.loc28_19.2 => constants.%y.patt.d967f4.2
// CHECK:STDOUT:   %.loc28_45.1 => constants.%.e9daaa.2
// CHECK:STDOUT:   %.loc28_45.2 => constants.%.237907.4
// CHECK:STDOUT:   %.loc28_45.3 => constants.%.f9658d.4
// CHECK:STDOUT:   %.loc28_47.1 => constants.%.6c4209.2
// CHECK:STDOUT:   %pattern_type.loc28_47 => constants.%pattern_type.940805.4
// CHECK:STDOUT:   %return.param_patt.loc28_47.2 => constants.%return.param_patt.bcf038.2
// CHECK:STDOUT:   %return.patt.loc28_34.2 => constants.%return.patt.690db9.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EnclosingTemplate(constants.%S) {
// CHECK:STDOUT:   %T.patt.loc26_32.2 => constants.%T.patt.3a0592.2
// CHECK:STDOUT:   %T.loc26_32.1 => constants.%S
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a55
// CHECK:STDOUT:   %a.patt.loc27_8.2 => constants.%a.patt.2a2
// CHECK:STDOUT:   %.loc27_15.3 => constants.%inst.name_ref.426619.1
// CHECK:STDOUT:   %.loc27_15.4 => constants.%S.Make.type
// CHECK:STDOUT:   %.loc27_15.5 => constants.%S.Make
// CHECK:STDOUT:   %.loc27_21.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc27_21.5 => constants.%S
// CHECK:STDOUT:   %.loc27_21.6 => <not constant>
// CHECK:STDOUT:   %.loc27_21.7 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc27_21.8 => <not constant>
// CHECK:STDOUT:   %Local.type => constants.%Local.type.2cd
// CHECK:STDOUT:   %Local => constants.%Local.6cd
// CHECK:STDOUT:   %.loc31_17.3 => constants.%inst.name_ref.426619.2
// CHECK:STDOUT:   %.loc31_17.4 => constants.%S.Make.type
// CHECK:STDOUT:   %.loc31_17.5 => constants.%S.Make
// CHECK:STDOUT:   %.loc31_23.3 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc31_23.4 => constants.%S
// CHECK:STDOUT:   %.loc31_23.5 => <not constant>
// CHECK:STDOUT:   %.loc31_27.3 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc31_27.4 => constants.%i32
// CHECK:STDOUT:   %.loc31_27.5 => <not constant>
// CHECK:STDOUT:   %.loc31_29.3 => constants.%inst.splice_block.426619.7
// CHECK:STDOUT:   %.loc31_29.4 => constants.%i32
// CHECK:STDOUT:   %.loc31_29.5 => <not constant>
// CHECK:STDOUT:   %.loc31_30.2 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc31_30.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Local.loc28(constants.%S) {
// CHECK:STDOUT:   %T => constants.%S
// CHECK:STDOUT:   %pattern_type.loc28_13 => constants.%pattern_type.a55
// CHECK:STDOUT:   %_.param_patt.loc28_13.2 => constants.%_.param_patt.230
// CHECK:STDOUT:   %_.patt.loc28_13.2 => constants.%_.patt.e4c
// CHECK:STDOUT:   %.loc28_29.1 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc28_29.2 => constants.%i32
// CHECK:STDOUT:   %.loc28_29.3 => <not constant>
// CHECK:STDOUT:   %pattern_type.loc28_19 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %y.param_patt.loc28_19.2 => constants.%y.param_patt.673
// CHECK:STDOUT:   %y.patt.loc28_19.2 => constants.%y.patt.547
// CHECK:STDOUT:   %.loc28_45.1 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc28_45.2 => constants.%i32
// CHECK:STDOUT:   %.loc28_45.3 => <not constant>
// CHECK:STDOUT:   %.loc28_47.1 => constants.%.795f
// CHECK:STDOUT:   %pattern_type.loc28_47 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc28_47.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc28_34.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_13 => constants.%complete_type.cdf
// CHECK:STDOUT:   %require_complete.loc28_19 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc28_34 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc29_13.2 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.9
// CHECK:STDOUT:   %.loc29_13.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
