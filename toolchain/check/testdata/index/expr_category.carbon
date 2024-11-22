// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/expr_category.carbon

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var pa: i32* = &a[0];
  a[0] = 4;
}

fn ValueBinding(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Index but don't do anything else so we can check that a value binding is
  // produced when appropriate.
  a[0];
  b[0];
  F()[0];
}

// CHECK:STDOUT: --- expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.34, %.37, %.40) [template]
// CHECK:STDOUT:   %.41: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.45: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.45, %Convert.14 [template]
// CHECK:STDOUT:   %.47: <specific function> = specific_function %.46, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.48: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [template]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .ValueBinding = %ValueBinding.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_12.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_12.3: type = converted %int.make_type_signed, %.loc11_12.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, %i32 [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.3 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_10.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_10.3: type = converted %int.make_type_signed.loc13, %.loc13_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15, %i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.3 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [template = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.3 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_21.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_26: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_21.2: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_21.3: type = converted %int.make_type_signed.loc21, %.loc21_21.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_27: type = array_type %.loc21_26, %i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.3 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_11.2: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_11.3: type = converted %int.make_type_signed.loc14, %.loc14_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17: type = array_type %.loc14_16, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_22: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_28: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_29.1: %tuple.type = tuple_literal (%.loc14_22, %.loc14_25, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_29.3: <bound method> = bound_method %.loc14_22, %.loc14_29.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.4: <specific function> = specific_function %.loc14_29.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.1: init %i32 = call %.loc14_29.4(%.loc14_22) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.5: init %i32 = converted %.loc14_22, %int.convert_checked.loc14_29.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.6: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_29.7: ref %i32 = array_index %a.var, %.loc14_29.6
// CHECK:STDOUT:   %.loc14_29.8: init %i32 = initialize_from %.loc14_29.5 to %.loc14_29.7 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.9: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_29.10: <bound method> = bound_method %.loc14_25, %.loc14_29.9 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_29.11: <specific function> = specific_function %.loc14_29.10, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.2: init %i32 = call %.loc14_29.11(%.loc14_25) [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_29.12: init %i32 = converted %.loc14_25, %int.convert_checked.loc14_29.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_29.13: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_29.14: ref %i32 = array_index %a.var, %.loc14_29.13
// CHECK:STDOUT:   %.loc14_29.15: init %i32 = initialize_from %.loc14_29.12 to %.loc14_29.14 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_29.16: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_29.17: <bound method> = bound_method %.loc14_28, %.loc14_29.16 [template = constants.%.38]
// CHECK:STDOUT:   %.loc14_29.18: <specific function> = specific_function %.loc14_29.17, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.3: init %i32 = call %.loc14_29.18(%.loc14_28) [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_29.19: init %i32 = converted %.loc14_28, %int.convert_checked.loc14_29.3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_29.20: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_29.21: ref %i32 = array_index %a.var, %.loc14_29.20
// CHECK:STDOUT:   %.loc14_29.22: init %i32 = initialize_from %.loc14_29.19 to %.loc14_29.21 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_29.23: init %.3 = array_init (%.loc14_29.8, %.loc14_29.15, %.loc14_29.22) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_30: init %.3 = converted %.loc14_29.1, %.loc14_29.23 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc14_30
// CHECK:STDOUT:   %.loc17_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_14.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_14.2: type = converted %int.make_type_signed.loc17, %.loc17_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_14.3: type = ptr_type %i32 [template = constants.%.41]
// CHECK:STDOUT:   %pa.var: ref %.41 = var pa
// CHECK:STDOUT:   %pa: ref %.41 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc17: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc17_21.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_21.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_21.3: <bound method> = bound_method %.loc17_21.1, %.loc17_21.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc17_21.4: <specific function> = specific_function %.loc17_21.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %.loc17_21.4(%.loc17_21.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc17_21.5: %i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%.44]
// CHECK:STDOUT:   %.loc17_21.6: %i32 = converted %.loc17_21.1, %.loc17_21.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc17_22: ref %i32 = array_index %a.ref.loc17, %.loc17_21.6
// CHECK:STDOUT:   %.loc17_18: %.41 = addr_of %.loc17_22
// CHECK:STDOUT:   assign %pa.var, %.loc17_18
// CHECK:STDOUT:   %a.ref.loc18: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_5.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_5.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_5.3: <bound method> = bound_method %.loc18_5.1, %.loc18_5.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc18_5.4: <specific function> = specific_function %.loc18_5.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc18_5: init %i32 = call %.loc18_5.4(%.loc18_5.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc18_5.5: %i32 = value_of_initializer %int.convert_checked.loc18_5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc18_5.6: %i32 = converted %.loc18_5.1, %.loc18_5.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc18_6: ref %i32 = array_index %a.ref.loc18, %.loc18_5.6
// CHECK:STDOUT:   %.loc18_10: Core.IntLiteral = int_value 4 [template = constants.%.45]
// CHECK:STDOUT:   %.loc18_8.1: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_8.2: <bound method> = bound_method %.loc18_10, %.loc18_8.1 [template = constants.%.46]
// CHECK:STDOUT:   %.loc18_8.3: <specific function> = specific_function %.loc18_8.2, @Convert.2(constants.%.1) [template = constants.%.47]
// CHECK:STDOUT:   %int.convert_checked.loc18_8: init %i32 = call %.loc18_8.3(%.loc18_10) [template = constants.%.48]
// CHECK:STDOUT:   %.loc18_8.4: init %i32 = converted %.loc18_10, %int.convert_checked.loc18_8 [template = constants.%.48]
// CHECK:STDOUT:   assign %.loc18_6, %.loc18_8.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_16: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_11.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_11.3: type = converted %int.make_type_signed.loc22, %.loc22_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_17: type = array_type %.loc22_16, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_22: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_25: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_28: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_29.1: %tuple.type = tuple_literal (%.loc22_22, %.loc22_25, %.loc22_28)
// CHECK:STDOUT:   %.loc22_29.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_29.3: <bound method> = bound_method %.loc22_22, %.loc22_29.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.4: <specific function> = specific_function %.loc22_29.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.1: init %i32 = call %.loc22_29.4(%.loc22_22) [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.5: init %i32 = converted %.loc22_22, %int.convert_checked.loc22_29.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.6: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc22_29.7: ref %i32 = array_index %a.var, %.loc22_29.6
// CHECK:STDOUT:   %.loc22_29.8: init %i32 = initialize_from %.loc22_29.5 to %.loc22_29.7 [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.9: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_29.10: <bound method> = bound_method %.loc22_25, %.loc22_29.9 [template = constants.%.35]
// CHECK:STDOUT:   %.loc22_29.11: <specific function> = specific_function %.loc22_29.10, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.2: init %i32 = call %.loc22_29.11(%.loc22_25) [template = constants.%.37]
// CHECK:STDOUT:   %.loc22_29.12: init %i32 = converted %.loc22_25, %int.convert_checked.loc22_29.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc22_29.13: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_29.14: ref %i32 = array_index %a.var, %.loc22_29.13
// CHECK:STDOUT:   %.loc22_29.15: init %i32 = initialize_from %.loc22_29.12 to %.loc22_29.14 [template = constants.%.37]
// CHECK:STDOUT:   %.loc22_29.16: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_29.17: <bound method> = bound_method %.loc22_28, %.loc22_29.16 [template = constants.%.38]
// CHECK:STDOUT:   %.loc22_29.18: <specific function> = specific_function %.loc22_29.17, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.3: init %i32 = call %.loc22_29.18(%.loc22_28) [template = constants.%.40]
// CHECK:STDOUT:   %.loc22_29.19: init %i32 = converted %.loc22_28, %int.convert_checked.loc22_29.3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc22_29.20: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_29.21: ref %i32 = array_index %a.var, %.loc22_29.20
// CHECK:STDOUT:   %.loc22_29.22: init %i32 = initialize_from %.loc22_29.19 to %.loc22_29.21 [template = constants.%.40]
// CHECK:STDOUT:   %.loc22_29.23: init %.3 = array_init (%.loc22_29.8, %.loc22_29.15, %.loc22_29.22) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc22_30: init %.3 = converted %.loc22_29.1, %.loc22_29.23 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc22_30
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_5.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_5.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_5.3: <bound method> = bound_method %.loc26_5.1, %.loc26_5.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc26_5.4: <specific function> = specific_function %.loc26_5.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %.loc26_5.4(%.loc26_5.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc26_5.5: %i32 = value_of_initializer %int.convert_checked.loc26 [template = constants.%.44]
// CHECK:STDOUT:   %.loc26_5.6: %i32 = converted %.loc26_5.1, %.loc26_5.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc26_6: ref %i32 = array_index %a.ref, %.loc26_5.6
// CHECK:STDOUT:   %b.ref: %.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc27_5.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc27_5.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc27_5.3: <bound method> = bound_method %.loc27_5.1, %.loc27_5.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc27_5.4: <specific function> = specific_function %.loc27_5.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %.loc27_5.4(%.loc27_5.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc27_5.5: %i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%.44]
// CHECK:STDOUT:   %.loc27_5.6: %i32 = converted %.loc27_5.1, %.loc27_5.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc27_6.1: ref %.3 = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc27_6.2: ref %i32 = array_index %.loc27_6.1, %.loc27_5.6
// CHECK:STDOUT:   %.loc27_6.3: %i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc28_4.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %.3 = call %F.ref() to %.loc28_4.1
// CHECK:STDOUT:   %.loc28_7.1: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc28_4.2: ref %.3 = temporary %.loc28_4.1, %F.call
// CHECK:STDOUT:   %.loc28_7.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc28_7.3: <bound method> = bound_method %.loc28_7.1, %.loc28_7.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc28_7.4: <specific function> = specific_function %.loc28_7.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc28: init %i32 = call %.loc28_7.4(%.loc28_7.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc28_7.5: %i32 = value_of_initializer %int.convert_checked.loc28 [template = constants.%.44]
// CHECK:STDOUT:   %.loc28_7.6: %i32 = converted %.loc28_7.1, %.loc28_7.5 [template = constants.%.44]
// CHECK:STDOUT:   %.loc28_8.1: ref %i32 = array_index %.loc28_4.2, %.loc28_7.6
// CHECK:STDOUT:   %.loc28_8.2: %i32 = bind_value %.loc28_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
