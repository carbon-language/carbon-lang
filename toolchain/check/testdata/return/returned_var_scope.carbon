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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %UnrelatedScopes.type: type = fn_type @UnrelatedScopes [template]
// CHECK:STDOUT:   %UnrelatedScopes: %UnrelatedScopes.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.28: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %EnclosingButAfter.type: type = fn_type @EnclosingButAfter [template]
// CHECK:STDOUT:   %EnclosingButAfter: %EnclosingButAfter.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.51
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
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_32.loc11, %.loc11_25.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EnclosingButAfter.decl: %EnclosingButAfter.type = fn_decl @EnclosingButAfter [template = constants.%EnclosingButAfter] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc21_25.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc21_25.2: type = converted %bool.make_type, %.loc21_25.1 [template = bool]
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_34.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_34.2: type = converted %int.make_type_32.loc21, %.loc21_34.1 [template = i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_32.loc13, %.loc13_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc13_27: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_28.2: <bound method> = bound_method %.loc13_27, %.loc13_28.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc13: init i32 = call %.loc13_28.2(%.loc13_27) [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_28.3: init i32 = converted %.loc13_27, %int.convert_checked.loc13 [template = constants.%.28]
// CHECK:STDOUT:   assign %v.var, %.loc13_28.3
// CHECK:STDOUT:   br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %.loc15: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc15 br !if.then.loc15 else br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc15:
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_21.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_21.2: type = converted %int.make_type_32.loc16, %.loc16_21.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc16_27: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc16_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_28.2: <bound method> = bound_method %.loc16_27, %.loc16_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc16: init i32 = call %.loc16_28.2(%.loc16_27) [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_28.3: init i32 = converted %.loc16_27, %int.convert_checked.loc16 [template = constants.%.31]
// CHECK:STDOUT:   assign %w.var, %.loc16_28.3
// CHECK:STDOUT:   br !if.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc15:
// CHECK:STDOUT:   %.loc18_10: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_11.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc18_11.2: <bound method> = bound_method %.loc18_10, %.loc18_11.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc18: init i32 = call %.loc18_11.2(%.loc18_10) [template = constants.%.28]
// CHECK:STDOUT:   %.loc18_11.3: i32 = value_of_initializer %int.convert_checked.loc18 [template = constants.%.28]
// CHECK:STDOUT:   %.loc18_11.4: i32 = converted %.loc18_10, %.loc18_11.3 [template = constants.%.28]
// CHECK:STDOUT:   return %.loc18_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b.param_patt: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_21.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_21.2: type = converted %int.make_type_32.loc23, %.loc23_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23_27: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc23_28.2: <bound method> = bound_method %.loc23_27, %.loc23_28.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc23: init i32 = call %.loc23_28.2(%.loc23_27) [template = constants.%.28]
// CHECK:STDOUT:   %.loc23_28.3: init i32 = converted %.loc23_27, %int.convert_checked.loc23 [template = constants.%.28]
// CHECK:STDOUT:   assign %v.var, %.loc23_28.3
// CHECK:STDOUT:   %.loc23_18: i32 = bind_value %v
// CHECK:STDOUT:   return %.loc23_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_19.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:   %.loc26_19.2: type = converted %int.make_type_32.loc26, %.loc26_19.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc26_25: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc26_26.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc26_26.2: <bound method> = bound_method %.loc26_25, %.loc26_26.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc26: init i32 = call %.loc26_26.2(%.loc26_25) [template = constants.%.31]
// CHECK:STDOUT:   %.loc26_26.3: init i32 = converted %.loc26_25, %int.convert_checked.loc26 [template = constants.%.31]
// CHECK:STDOUT:   assign %w.var, %.loc26_26.3
// CHECK:STDOUT:   %.loc26_16: i32 = bind_value %w
// CHECK:STDOUT:   return %.loc26_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
