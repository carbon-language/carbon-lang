// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_shadow.carbon

fn SameScope() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+7]]:18: error: cannot declare a `returned var` in the scope of another `returned var` [ReturnedVarShadowed]
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-4]]:18: note: `returned var` was declared here [ReturnedVarHere]
    // CHECK:STDERR:     returned var v: i32 = 0;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR:
    returned var w: i32 = 1;
  }
  return 0;
}

fn DifferentScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    if (true) {
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+6]]:20: error: cannot declare a `returned var` in the scope of another `returned var` [ReturnedVarShadowed]
      // CHECK:STDERR:       returned var w: i32 = 1;
      // CHECK:STDERR:                    ^
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-5]]:18: note: `returned var` was declared here [ReturnedVarHere]
      // CHECK:STDERR:     returned var v: i32 = 0;
      // CHECK:STDERR:                  ^
      returned var w: i32 = 1;
    }
  }
  return 0;
}

// CHECK:STDOUT: --- fail_returned_var_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %SameScope.type: type = fn_type @SameScope [template]
// CHECK:STDOUT:   %SameScope: %SameScope.type = struct_value () [template]
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
// CHECK:STDOUT:   %DifferentScopes.type: type = fn_type @DifferentScopes [template]
// CHECK:STDOUT:   %DifferentScopes: %DifferentScopes.type = struct_value () [template]
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
// CHECK:STDOUT:     .SameScope = %SameScope.decl
// CHECK:STDOUT:     .DifferentScopes = %DifferentScopes.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SameScope.decl: %SameScope.type = fn_decl @SameScope [template = constants.%SameScope] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentScopes.decl: %DifferentScopes.type = fn_decl @DifferentScopes [template = constants.%DifferentScopes] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc26_25.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:     %.loc26_25.2: type = converted %int.make_type_32.loc26, %.loc26_25.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc12 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
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
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_21.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_21.2: type = converted %int.make_type_32.loc21, %.loc21_21.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc21_27: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc21_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc21_28.2: <bound method> = bound_method %.loc21_27, %.loc21_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc21: init i32 = call %.loc21_28.2(%.loc21_27) [template = constants.%.31]
// CHECK:STDOUT:   %.loc21_28.3: init i32 = converted %.loc21_27, %int.convert_checked.loc21 [template = constants.%.31]
// CHECK:STDOUT:   assign %w.var, %.loc21_28.3
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc23_10: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_11.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc23_11.2: <bound method> = bound_method %.loc23_10, %.loc23_11.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc23: init i32 = call %.loc23_11.2(%.loc23_10) [template = constants.%.28]
// CHECK:STDOUT:   %.loc23_11.3: i32 = value_of_initializer %int.convert_checked.loc23 [template = constants.%.28]
// CHECK:STDOUT:   %.loc23_11.4: i32 = converted %.loc23_10, %.loc23_11.3 [template = constants.%.28]
// CHECK:STDOUT:   return %.loc23_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc27 br !if.then.loc27 else br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc27:
// CHECK:STDOUT:   %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc28_21.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:   %.loc28_21.2: type = converted %int.make_type_32.loc28, %.loc28_21.1 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc28_27: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_28.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc28_28.2: <bound method> = bound_method %.loc28_27, %.loc28_28.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc28: init i32 = call %.loc28_28.2(%.loc28_27) [template = constants.%.28]
// CHECK:STDOUT:   %.loc28_28.3: init i32 = converted %.loc28_27, %int.convert_checked.loc28 [template = constants.%.28]
// CHECK:STDOUT:   assign %v.var, %.loc28_28.3
// CHECK:STDOUT:   %.loc29: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc29 br !if.then.loc29 else br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc29:
// CHECK:STDOUT:   %int.make_type_32.loc36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc36_23.1: type = value_of_initializer %int.make_type_32.loc36 [template = i32]
// CHECK:STDOUT:   %.loc36_23.2: type = converted %int.make_type_32.loc36, %.loc36_23.1 [template = i32]
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc36_29: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc36_30.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc36_30.2: <bound method> = bound_method %.loc36_29, %.loc36_30.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc36: init i32 = call %.loc36_30.2(%.loc36_29) [template = constants.%.31]
// CHECK:STDOUT:   %.loc36_30.3: init i32 = converted %.loc36_29, %int.convert_checked.loc36 [template = constants.%.31]
// CHECK:STDOUT:   assign %w.var, %.loc36_30.3
// CHECK:STDOUT:   br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc29:
// CHECK:STDOUT:   br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc27:
// CHECK:STDOUT:   %.loc39_10: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc39_11.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc39_11.2: <bound method> = bound_method %.loc39_10, %.loc39_11.1 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc39: init i32 = call %.loc39_11.2(%.loc39_10) [template = constants.%.28]
// CHECK:STDOUT:   %.loc39_11.3: i32 = value_of_initializer %int.convert_checked.loc39 [template = constants.%.28]
// CHECK:STDOUT:   %.loc39_11.4: i32 = converted %.loc39_10, %.loc39_11.3 [template = constants.%.28]
// CHECK:STDOUT:   return %.loc39_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
