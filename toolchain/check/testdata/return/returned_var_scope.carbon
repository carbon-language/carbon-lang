// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/returned_var_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/returned_var_scope.carbon

fn UnrelatedScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  if (true) {
    returned var w: i32 = 1;
  }
  return 0;
}

fn EnclosingButAfter(b: bool) -> i32 {
  if (b) {
    returned var v: i32 = 0;
    return var;
  }
  returned var w: i32 = 1;
  return var;
}

// CHECK:STDOUT: --- returned_var_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %UnrelatedScopes.type: type = fn_type @UnrelatedScopes [template]
// CHECK:STDOUT:   %UnrelatedScopes: %UnrelatedScopes.type = struct_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal true [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %EnclosingButAfter.type: type = fn_type @EnclosingButAfter [template]
// CHECK:STDOUT:   %EnclosingButAfter: %EnclosingButAfter.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .UnrelatedScopes = %UnrelatedScopes.decl
// CHECK:STDOUT:     .EnclosingButAfter = %EnclosingButAfter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %UnrelatedScopes.decl: %UnrelatedScopes.type = fn_decl @UnrelatedScopes [template = constants.%UnrelatedScopes] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_25.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_signed.loc11, %.loc11_25.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EnclosingButAfter.decl: %EnclosingButAfter.type = fn_decl @EnclosingButAfter [template = constants.%EnclosingButAfter] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc21_25.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc21_25.2: type = converted %bool.make_type, %.loc21_25.1 [template = bool]
// CHECK:STDOUT:     %.loc21_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_34.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_34.2: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_34.3: type = converted %int.make_type_signed.loc21, %.loc21_34.2 [template = constants.%.2]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: bool = bool_literal true [template = constants.%.3]
// CHECK:STDOUT:   if %.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc13_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_21.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_21.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_21.3: type = converted %int.make_type_signed.loc13, %.loc13_21.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.var: ref %.2 = var v
// CHECK:STDOUT:   %v: ref %.2 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc13_27: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_28.2: <bound method> = bound_method %.loc13_27, %.loc13_28.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_28.3: <specific function> = specific_function %.loc13_28.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %.2 = call %.loc13_28.3(%.loc13_27) [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_28.4: init %.2 = converted %.loc13_27, %int.convert_checked.loc13 [template = constants.%.29]
// CHECK:STDOUT:   assign %v.var, %.loc13_28.4
// CHECK:STDOUT:   br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %.loc15: bool = bool_literal true [template = constants.%.3]
// CHECK:STDOUT:   if %.loc15 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %.loc16_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%.loc16_21.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_21.2: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_21.3: type = converted %int.make_type_signed.loc16, %.loc16_21.2 [template = constants.%.2]
// CHECK:STDOUT:   %w.var: ref %.2 = var w
// CHECK:STDOUT:   %w: ref %.2 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc16_27: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_28.2: <bound method> = bound_method %.loc16_27, %.loc16_28.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_28.3: <specific function> = specific_function %.loc16_28.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %.2 = call %.loc16_28.3(%.loc16_27) [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_28.4: init %.2 = converted %.loc16_27, %int.convert_checked.loc16 [template = constants.%.33]
// CHECK:STDOUT:   assign %w.var, %.loc16_28.4
// CHECK:STDOUT:   br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %.loc18_10: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_11.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_11.2: <bound method> = bound_method %.loc18_10, %.loc18_11.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc18_11.3: <specific function> = specific_function %.loc18_11.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %.2 = call %.loc18_11.3(%.loc18_10) [template = constants.%.29]
// CHECK:STDOUT:   %.loc18_11.4: %.2 = value_of_initializer %int.convert_checked.loc18 [template = constants.%.29]
// CHECK:STDOUT:   %.loc18_11.5: %.2 = converted %.loc18_10, %.loc18_11.4 [template = constants.%.29]
// CHECK:STDOUT:   return %.loc18_11.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b.param_patt: bool) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc23_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%.loc23_21.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_21.2: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_21.3: type = converted %int.make_type_signed.loc23, %.loc23_21.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.var: ref %.2 = var v
// CHECK:STDOUT:   %v: ref %.2 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23_27: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc23_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc23_28.2: <bound method> = bound_method %.loc23_27, %.loc23_28.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc23_28.3: <specific function> = specific_function %.loc23_28.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %.2 = call %.loc23_28.3(%.loc23_27) [template = constants.%.29]
// CHECK:STDOUT:   %.loc23_28.4: init %.2 = converted %.loc23_27, %int.convert_checked.loc23 [template = constants.%.29]
// CHECK:STDOUT:   assign %v.var, %.loc23_28.4
// CHECK:STDOUT:   %.loc23_18: %.2 = bind_value %v
// CHECK:STDOUT:   return %.loc23_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc26_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc26: init type = call constants.%Int(%.loc26_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc26_19.2: type = value_of_initializer %int.make_type_signed.loc26 [template = constants.%.2]
// CHECK:STDOUT:   %.loc26_19.3: type = converted %int.make_type_signed.loc26, %.loc26_19.2 [template = constants.%.2]
// CHECK:STDOUT:   %w.var: ref %.2 = var w
// CHECK:STDOUT:   %w: ref %.2 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc26_25: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc26_26.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_26.2: <bound method> = bound_method %.loc26_25, %.loc26_26.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc26_26.3: <specific function> = specific_function %.loc26_26.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %.2 = call %.loc26_26.3(%.loc26_25) [template = constants.%.33]
// CHECK:STDOUT:   %.loc26_26.4: init %.2 = converted %.loc26_25, %int.convert_checked.loc26 [template = constants.%.33]
// CHECK:STDOUT:   assign %w.var, %.loc26_26.4
// CHECK:STDOUT:   %.loc26_16: %.2 = bind_value %w
// CHECK:STDOUT:   return %.loc26_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
