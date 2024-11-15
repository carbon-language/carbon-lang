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
// CHECK:STDOUT:   %.1: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.32: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.28 = tuple_value (%.30, %.31, %.1) [template]
// CHECK:STDOUT:   %.33: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 4 [template]
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
// CHECK:STDOUT:     %return.patt: %.28 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.28 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_17.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc11_17.3: <bound method> = bound_method %.loc11_17.1, %.loc11_17.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %.loc11_17.3(%.loc11_17.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_17.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_17.5: Core.IntLiteral = converted %.loc11_17.1, %.loc11_17.4 [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17.5, i32 [template = constants.%.28]
// CHECK:STDOUT:     %return.param: ref %.28 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.28 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.28 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.28 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_15.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:     %.loc13_15.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc13_15.3: <bound method> = bound_method %.loc13_15.1, %.loc13_15.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc13: init Core.IntLiteral = call %.loc13_15.3(%.loc13_15.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_15.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13 [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_15.5: Core.IntLiteral = converted %.loc13_15.1, %.loc13_15.4 [template = constants.%.27]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15.5, i32 [template = constants.%.28]
// CHECK:STDOUT:     %b.param: %.28 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.28 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [template = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %.28 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.28 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_26.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc21_21.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_21.2: type = converted %int.make_type_32.loc21, %.loc21_21.1 [template = i32]
// CHECK:STDOUT:     %.loc21_26.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc21_26.3: <bound method> = bound_method %.loc21_26.1, %.loc21_26.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked.loc21: init Core.IntLiteral = call %.loc21_26.3(%.loc21_26.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc21_26.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc21 [template = constants.%.27]
// CHECK:STDOUT:     %.loc21_26.5: Core.IntLiteral = converted %.loc21_26.1, %.loc21_26.4 [template = constants.%.27]
// CHECK:STDOUT:     %.loc21_27: type = array_type %.loc21_26.5, i32 [template = constants.%.28]
// CHECK:STDOUT:     %b.param: %.28 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.28 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.28;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.28) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_16.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_11.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_11.2: type = converted %int.make_type_32.loc14, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:   %.loc14_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_16.3: <bound method> = bound_method %.loc14_16.1, %.loc14_16.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc14: init Core.IntLiteral = call %.loc14_16.3(%.loc14_16.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_16.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_16.5: Core.IntLiteral = converted %.loc14_16.1, %.loc14_16.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_17: type = array_type %.loc14_16.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %a.var: ref %.28 = var a
// CHECK:STDOUT:   %a: ref %.28 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_22: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_25: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_28: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_29.1: %tuple.type = tuple_literal (%.loc14_22, %.loc14_25, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.2: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_29.3: ref i32 = array_index %a.var, %.loc14_29.2
// CHECK:STDOUT:   %.loc14_29.4: init i32 = initialize_from %.loc14_22 to %.loc14_29.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_29.5: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_29.6: ref i32 = array_index %a.var, %.loc14_29.5
// CHECK:STDOUT:   %.loc14_29.7: init i32 = initialize_from %.loc14_25 to %.loc14_29.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_29.8: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_29.9: ref i32 = array_index %a.var, %.loc14_29.8
// CHECK:STDOUT:   %.loc14_29.10: init i32 = initialize_from %.loc14_28 to %.loc14_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_29.11: init %.28 = array_init (%.loc14_29.4, %.loc14_29.7, %.loc14_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_30: init %.28 = converted %.loc14_29.1, %.loc14_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc14_30
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_14.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_14.2: type = converted %int.make_type_32.loc17, %.loc17_14.1 [template = i32]
// CHECK:STDOUT:   %.loc17_14.3: type = ptr_type i32 [template = constants.%.33]
// CHECK:STDOUT:   %pa.var: ref %.33 = var pa
// CHECK:STDOUT:   %pa: ref %.33 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc17: ref %.28 = name_ref a, %a
// CHECK:STDOUT:   %.loc17_21: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc17_22: ref i32 = array_index %a.ref.loc17, %.loc17_21
// CHECK:STDOUT:   %.loc17_18: %.33 = addr_of %.loc17_22
// CHECK:STDOUT:   assign %pa.var, %.loc17_18
// CHECK:STDOUT:   %a.ref.loc18: ref %.28 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_5: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc18_6: ref i32 = array_index %a.ref.loc18, %.loc18_5
// CHECK:STDOUT:   %.loc18_10: i32 = int_value 4 [template = constants.%.34]
// CHECK:STDOUT:   assign %.loc18_6, %.loc18_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param_patt: %.28) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_16.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_11.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %int.make_type_32.loc22, %.loc22_11.1 [template = i32]
// CHECK:STDOUT:   %.loc22_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc22_16.3: <bound method> = bound_method %.loc22_16.1, %.loc22_16.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc22: init Core.IntLiteral = call %.loc22_16.3(%.loc22_16.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_16.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc22 [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_16.5: Core.IntLiteral = converted %.loc22_16.1, %.loc22_16.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_17: type = array_type %.loc22_16.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %a.var: ref %.28 = var a
// CHECK:STDOUT:   %a: ref %.28 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_22: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_25: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc22_28: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_29.1: %tuple.type = tuple_literal (%.loc22_22, %.loc22_25, %.loc22_28)
// CHECK:STDOUT:   %.loc22_29.2: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc22_29.3: ref i32 = array_index %a.var, %.loc22_29.2
// CHECK:STDOUT:   %.loc22_29.4: init i32 = initialize_from %.loc22_22 to %.loc22_29.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_29.5: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_29.6: ref i32 = array_index %a.var, %.loc22_29.5
// CHECK:STDOUT:   %.loc22_29.7: init i32 = initialize_from %.loc22_25 to %.loc22_29.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc22_29.8: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc22_29.9: ref i32 = array_index %a.var, %.loc22_29.8
// CHECK:STDOUT:   %.loc22_29.10: init i32 = initialize_from %.loc22_28 to %.loc22_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_29.11: init %.28 = array_init (%.loc22_29.4, %.loc22_29.7, %.loc22_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc22_30: init %.28 = converted %.loc22_29.1, %.loc22_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc22_30
// CHECK:STDOUT:   %a.ref: ref %.28 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_5: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc26_6: ref i32 = array_index %a.ref, %.loc26_5
// CHECK:STDOUT:   %b.ref: %.28 = name_ref b, %b
// CHECK:STDOUT:   %.loc27_5: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc27_6.1: ref %.28 = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc27_6.2: ref i32 = array_index %.loc27_6.1, %.loc27_5
// CHECK:STDOUT:   %.loc27_6.3: i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc28_4.1: ref %.28 = temporary_storage
// CHECK:STDOUT:   %F.call: init %.28 = call %F.ref() to %.loc28_4.1
// CHECK:STDOUT:   %.loc28_7: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc28_4.2: ref %.28 = temporary %.loc28_4.1, %F.call
// CHECK:STDOUT:   %.loc28_8.1: ref i32 = array_index %.loc28_4.2, %.loc28_7
// CHECK:STDOUT:   %.loc28_8.2: i32 = bind_value %.loc28_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
