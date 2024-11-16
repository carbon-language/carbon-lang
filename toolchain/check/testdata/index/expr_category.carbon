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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.2 = tuple_value (%.32, %.34, %.36) [template]
// CHECK:STDOUT:   %.37: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.38, %Convert.15 [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.40, %Convert.15 [template]
// CHECK:STDOUT:   %.42: i32 = int_value 4 [template]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [template]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
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
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, i32 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.2 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_15: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15, i32 [template = constants.%.2]
// CHECK:STDOUT:     %b.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.2 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [template = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.2 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_26: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc21_21.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_21.2: type = converted %int.make_type_32.loc21, %.loc21_21.1 [template = i32]
// CHECK:STDOUT:     %.loc21_27: type = array_type %.loc21_26, i32 [template = constants.%.2]
// CHECK:STDOUT:     %b.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.2 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_11.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_11.2: type = converted %int.make_type_32.loc14, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:   %.loc14_17: type = array_type %.loc14_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_22: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_25: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_28: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_29.1: %tuple.type = tuple_literal (%.loc14_22, %.loc14_25, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_29.3: <bound method> = bound_method %.loc14_22, %.loc14_29.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.1: init i32 = call %.loc14_29.3(%.loc14_22) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.4: init i32 = converted %.loc14_22, %int.convert_checked.loc14_29.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.5: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_29.6: ref i32 = array_index %a.var, %.loc14_29.5
// CHECK:STDOUT:   %.loc14_29.7: init i32 = initialize_from %.loc14_29.4 to %.loc14_29.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_29.9: <bound method> = bound_method %.loc14_25, %.loc14_29.8 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.2: init i32 = call %.loc14_29.9(%.loc14_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.10: init i32 = converted %.loc14_25, %int.convert_checked.loc14_29.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.11: i32 = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.12: ref i32 = array_index %a.var, %.loc14_29.11
// CHECK:STDOUT:   %.loc14_29.13: init i32 = initialize_from %.loc14_29.10 to %.loc14_29.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.14: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_29.15: <bound method> = bound_method %.loc14_28, %.loc14_29.14 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc14_29.3: init i32 = call %.loc14_29.15(%.loc14_28) [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_29.16: init i32 = converted %.loc14_28, %int.convert_checked.loc14_29.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_29.17: i32 = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_29.18: ref i32 = array_index %a.var, %.loc14_29.17
// CHECK:STDOUT:   %.loc14_29.19: init i32 = initialize_from %.loc14_29.16 to %.loc14_29.18 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_29.20: init %.2 = array_init (%.loc14_29.7, %.loc14_29.13, %.loc14_29.19) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_30: init %.2 = converted %.loc14_29.1, %.loc14_29.20 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc14_30
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_14.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_14.2: type = converted %int.make_type_32.loc17, %.loc17_14.1 [template = i32]
// CHECK:STDOUT:   %.loc17_14.3: type = ptr_type i32 [template = constants.%.37]
// CHECK:STDOUT:   %pa.var: ref %.37 = var pa
// CHECK:STDOUT:   %pa: ref %.37 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc17: ref %.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc17_21.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_21.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc17_21.3: <bound method> = bound_method %.loc17_21.1, %.loc17_21.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc17: init i32 = call %.loc17_21.3(%.loc17_21.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_21.4: i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_21.5: i32 = converted %.loc17_21.1, %.loc17_21.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_22: ref i32 = array_index %a.ref.loc17, %.loc17_21.5
// CHECK:STDOUT:   %.loc17_18: %.37 = addr_of %.loc17_22
// CHECK:STDOUT:   assign %pa.var, %.loc17_18
// CHECK:STDOUT:   %a.ref.loc18: ref %.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_5.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc18_5.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc18_5.3: <bound method> = bound_method %.loc18_5.1, %.loc18_5.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc18_5: init i32 = call %.loc18_5.3(%.loc18_5.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc18_5.4: i32 = value_of_initializer %int.convert_checked.loc18_5 [template = constants.%.6]
// CHECK:STDOUT:   %.loc18_5.5: i32 = converted %.loc18_5.1, %.loc18_5.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc18_6: ref i32 = array_index %a.ref.loc18, %.loc18_5.5
// CHECK:STDOUT:   %.loc18_10: Core.IntLiteral = int_value 4 [template = constants.%.40]
// CHECK:STDOUT:   %.loc18_8.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc18_8.2: <bound method> = bound_method %.loc18_10, %.loc18_8.1 [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc18_8: init i32 = call %.loc18_8.2(%.loc18_10) [template = constants.%.42]
// CHECK:STDOUT:   %.loc18_8.3: init i32 = converted %.loc18_10, %int.convert_checked.loc18_8 [template = constants.%.42]
// CHECK:STDOUT:   assign %.loc18_6, %.loc18_8.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param_patt: %.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_16: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_11.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %int.make_type_32.loc22, %.loc22_11.1 [template = i32]
// CHECK:STDOUT:   %.loc22_17: type = array_type %.loc22_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_22: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc22_25: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_28: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_29.1: %tuple.type = tuple_literal (%.loc22_22, %.loc22_25, %.loc22_28)
// CHECK:STDOUT:   %.loc22_29.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc22_29.3: <bound method> = bound_method %.loc22_22, %.loc22_29.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.1: init i32 = call %.loc22_29.3(%.loc22_22) [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.4: init i32 = converted %.loc22_22, %int.convert_checked.loc22_29.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.5: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_29.6: ref i32 = array_index %a.var, %.loc22_29.5
// CHECK:STDOUT:   %.loc22_29.7: init i32 = initialize_from %.loc22_29.4 to %.loc22_29.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc22_29.9: <bound method> = bound_method %.loc22_25, %.loc22_29.8 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.2: init i32 = call %.loc22_29.9(%.loc22_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.10: init i32 = converted %.loc22_25, %int.convert_checked.loc22_29.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.11: i32 = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.12: ref i32 = array_index %a.var, %.loc22_29.11
// CHECK:STDOUT:   %.loc22_29.13: init i32 = initialize_from %.loc22_29.10 to %.loc22_29.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.14: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc22_29.15: <bound method> = bound_method %.loc22_28, %.loc22_29.14 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc22_29.3: init i32 = call %.loc22_29.15(%.loc22_28) [template = constants.%.36]
// CHECK:STDOUT:   %.loc22_29.16: init i32 = converted %.loc22_28, %int.convert_checked.loc22_29.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc22_29.17: i32 = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc22_29.18: ref i32 = array_index %a.var, %.loc22_29.17
// CHECK:STDOUT:   %.loc22_29.19: init i32 = initialize_from %.loc22_29.16 to %.loc22_29.18 [template = constants.%.36]
// CHECK:STDOUT:   %.loc22_29.20: init %.2 = array_init (%.loc22_29.7, %.loc22_29.13, %.loc22_29.19) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc22_30: init %.2 = converted %.loc22_29.1, %.loc22_29.20 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc22_30
// CHECK:STDOUT:   %a.ref: ref %.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_5.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc26_5.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc26_5.3: <bound method> = bound_method %.loc26_5.1, %.loc26_5.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc26: init i32 = call %.loc26_5.3(%.loc26_5.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_5.4: i32 = value_of_initializer %int.convert_checked.loc26 [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_5.5: i32 = converted %.loc26_5.1, %.loc26_5.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_6: ref i32 = array_index %a.ref, %.loc26_5.5
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc27_5.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc27_5.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc27_5.3: <bound method> = bound_method %.loc27_5.1, %.loc27_5.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc27: init i32 = call %.loc27_5.3(%.loc27_5.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc27_5.4: i32 = value_of_initializer %int.convert_checked.loc27 [template = constants.%.6]
// CHECK:STDOUT:   %.loc27_5.5: i32 = converted %.loc27_5.1, %.loc27_5.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc27_6.1: ref %.2 = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc27_6.2: ref i32 = array_index %.loc27_6.1, %.loc27_5.5
// CHECK:STDOUT:   %.loc27_6.3: i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc28_4.1: ref %.2 = temporary_storage
// CHECK:STDOUT:   %F.call: init %.2 = call %F.ref() to %.loc28_4.1
// CHECK:STDOUT:   %.loc28_7.1: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc28_4.2: ref %.2 = temporary %.loc28_4.1, %F.call
// CHECK:STDOUT:   %.loc28_7.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc28_7.3: <bound method> = bound_method %.loc28_7.1, %.loc28_7.2 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc28: init i32 = call %.loc28_7.3(%.loc28_7.1) [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_7.4: i32 = value_of_initializer %int.convert_checked.loc28 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_7.5: i32 = converted %.loc28_7.1, %.loc28_7.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_8.1: ref i32 = array_index %.loc28_4.2, %.loc28_7.5
// CHECK:STDOUT:   %.loc28_8.2: i32 = bind_value %.loc28_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
