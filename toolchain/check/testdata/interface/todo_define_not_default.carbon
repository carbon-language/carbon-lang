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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/todo_define_not_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/todo_define_not_default.carbon

interface I {
  // TODO: A definition without `default` in an interface should be rejected.
  fn F() {}
  fn G(a: i32, b: i32) -> i32 = "int.sadd";

  // TODO: An associated constant with an initializer without `default` in an
  // interface should be rejected.
  let T:! type = (i32, i32);
  let N:! i32 = 42;
}

// CHECK:STDOUT: --- todo_define_not_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%G.decl [concrete]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.%T [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %assoc3: %I.assoc_type = assoc_entity element3, @I.%N [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.20e, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc18_11: type = splice_block %i32.loc18_11 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc18_19: type = splice_block %i32.loc18_19 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %G.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc2: %I.assoc_type = assoc_entity element2, @I.%T [concrete = constants.%assoc2]
// CHECK:STDOUT:     %int_32.loc22_19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc22_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22_27: %tuple.type.24b = tuple_literal (%i32.loc22_19, %i32.loc22_24)
// CHECK:STDOUT:     %.loc22_28: type = converted %.loc22_27, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc3: %I.assoc_type = assoc_entity element3, @I.%N [concrete = constants.%assoc3]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc23_19.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc23_19.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc23_19.2(%int_42) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:     %.loc23_19.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_42.c68]
// CHECK:STDOUT:     %.loc23_19.2: %i32 = converted %int_42, %.loc23_19.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   .T = @T.%assoc2
// CHECK:STDOUT:   .N = @N.%assoc3
// CHECK:STDOUT:   witness = (%F.decl, %G.decl, %T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const T:! type = %.loc22_28;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const N:! %i32 = %.loc23_19.2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: %i32, %b.param: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
