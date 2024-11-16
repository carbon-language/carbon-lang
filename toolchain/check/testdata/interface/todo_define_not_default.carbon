// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type, %F.type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %I.type, %G.type [template]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element1, @I.%G.decl [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%.4, %.4) [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element2, @I.%T [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.4) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.3) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.9, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.3) [template]
// CHECK:STDOUT:   %.34: %.4 = int_value 42 [template]
// CHECK:STDOUT:   %.35: type = assoc_entity_type %I.type, %.4 [template]
// CHECK:STDOUT:   %.36: %.35 = assoc_entity element3, @I.%N [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc13: %.1 = assoc_entity element0, %F.decl [template = constants.%.2]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %.4 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.4 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %.4 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.4 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_11.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc14_11: init type = call constants.%Int(%.loc14_11.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_11.2: type = value_of_initializer %int.make_type_signed.loc14_11 [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_11.3: type = converted %int.make_type_signed.loc14_11, %.loc14_11.2 [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_19.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc14_19: init type = call constants.%Int(%.loc14_19.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_19.2: type = value_of_initializer %int.make_type_signed.loc14_19 [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_19.3: type = converted %int.make_type_signed.loc14_19, %.loc14_19.2 [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_27.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc14_27: init type = call constants.%Int(%.loc14_27.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_27.2: type = value_of_initializer %int.make_type_signed.loc14_27 [template = constants.%.4]
// CHECK:STDOUT:     %.loc14_27.3: type = converted %int.make_type_signed.loc14_27, %.loc14_27.2 [template = constants.%.4]
// CHECK:STDOUT:     %a.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.4 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %.4 = value_param runtime_param1
// CHECK:STDOUT:     %b: %.4 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: %.5 = assoc_entity element1, %G.decl [template = constants.%.6]
// CHECK:STDOUT:   %.loc18_19: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc18_19: init type = call constants.%Int(%.loc18_19) [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_24: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc18_24: init type = call constants.%Int(%.loc18_24) [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_27: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc18_19, %int.make_type_signed.loc18_24)
// CHECK:STDOUT:   %.loc18_28.1: type = value_of_initializer %int.make_type_signed.loc18_19 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.2: type = converted %int.make_type_signed.loc18_19, %.loc18_28.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.3: type = value_of_initializer %int.make_type_signed.loc18_24 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.4: type = converted %int.make_type_signed.loc18_24, %.loc18_28.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.5: type = converted %.loc18_27, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc18_28.6: %.7 = assoc_entity element2, %T [template = constants.%.8]
// CHECK:STDOUT:   %.loc19_11.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_11.1) [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_11.2: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_11.3: type = converted %int.make_type_signed.loc19, %.loc19_11.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_17: Core.IntLiteral = int_value 42 [template = constants.%.9]
// CHECK:STDOUT:   %.loc19_19.1: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_19.2: <bound method> = bound_method %.loc19_17, %.loc19_19.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_19.3: <specific function> = specific_function %.loc19_19.2, @Convert.2(constants.%.3) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked: init %.4 = call %.loc19_19.3(%.loc19_17) [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_19.4: %.4 = value_of_initializer %int.convert_checked [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_19.5: %.4 = converted %.loc19_17, %.loc19_19.4 [template = constants.%.34]
// CHECK:STDOUT:   %N: %.4 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc19_19.6: %.35 = assoc_entity element3, %N [template = constants.%.36]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc13
// CHECK:STDOUT:   .G = %.loc14
// CHECK:STDOUT:   .T = %.loc18_28.6
// CHECK:STDOUT:   .N = %.loc19_19.6
// CHECK:STDOUT:   witness = (%F.decl, %G.decl, %T, %N)
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
// CHECK:STDOUT:   fn(%a.param_patt: %.4, %b.param_patt: %.4) -> %.4 = "int.sadd";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Self.1) {}
// CHECK:STDOUT:
