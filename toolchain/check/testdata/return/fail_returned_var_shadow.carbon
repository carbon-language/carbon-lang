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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %SameScope.type: type = fn_type @SameScope [template]
// CHECK:STDOUT:   %SameScope: %SameScope.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %int.0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %int.0.1, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %DifferentScopes.type: type = fn_type @DifferentScopes [template]
// CHECK:STDOUT:   %DifferentScopes: %DifferentScopes.type = struct_value () [template]
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
// CHECK:STDOUT:     .SameScope = %SameScope.decl
// CHECK:STDOUT:     .DifferentScopes = %DifferentScopes.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SameScope.decl: %SameScope.type = fn_decl @SameScope [template = constants.%SameScope] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_signed.loc11, %.loc11_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentScopes.decl: %DifferentScopes.type = fn_decl @DifferentScopes [template = constants.%DifferentScopes] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc26: init type = call constants.%Int(%int.32.loc26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_25.1: type = value_of_initializer %int.make_type_signed.loc26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_25.2: type = converted %int.make_type_signed.loc26, %.loc26_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc12 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int.32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_signed.loc13, %.loc13_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int.0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc13_28.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_28.2: <bound method> = bound_method %int.0.loc13, %.loc13_28.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc13_28.3: <specific function> = specific_function %.loc13_28.2, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %.loc13_28.3(%int.0.loc13) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc13_28.4: init %i32 = converted %int.0.loc13, %int.convert_checked.loc13 [template = constants.%int.0.2]
// CHECK:STDOUT:   assign %v.var, %.loc13_28.4
// CHECK:STDOUT:   %int.32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int.32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_21.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_21.2: type = converted %int.make_type_signed.loc21, %.loc21_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %w.var: ref %i32 = var w
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc21_28.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_28.2: <bound method> = bound_method %int.1, %.loc21_28.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc21_28.3: <specific function> = specific_function %.loc21_28.2, @Convert.2(constants.%int.32) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %.loc21_28.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc21_28.4: init %i32 = converted %int.1, %int.convert_checked.loc21 [template = constants.%int.1.2]
// CHECK:STDOUT:   assign %w.var, %.loc21_28.4
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int.0.loc23: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc23_11.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc23_11.2: <bound method> = bound_method %int.0.loc23, %.loc23_11.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc23_11.3: <specific function> = specific_function %.loc23_11.2, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %.loc23_11.3(%int.0.loc23) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc23_11.4: %i32 = value_of_initializer %int.convert_checked.loc23 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc23_11.5: %i32 = converted %int.0.loc23, %.loc23_11.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   return %.loc23_11.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc27 br !if.then.loc27 else br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc27:
// CHECK:STDOUT:   %int.32.loc28: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc28: init type = call constants.%Int(%int.32.loc28) [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_21.1: type = value_of_initializer %int.make_type_signed.loc28 [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_21.2: type = converted %int.make_type_signed.loc28, %.loc28_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int.0.loc28: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc28_28.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc28_28.2: <bound method> = bound_method %int.0.loc28, %.loc28_28.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc28_28.3: <specific function> = specific_function %.loc28_28.2, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc28: init %i32 = call %.loc28_28.3(%int.0.loc28) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc28_28.4: init %i32 = converted %int.0.loc28, %int.convert_checked.loc28 [template = constants.%int.0.2]
// CHECK:STDOUT:   assign %v.var, %.loc28_28.4
// CHECK:STDOUT:   %.loc29: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc29 br !if.then.loc29 else br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc29:
// CHECK:STDOUT:   %int.32.loc36: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc36: init type = call constants.%Int(%int.32.loc36) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_23.1: type = value_of_initializer %int.make_type_signed.loc36 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_23.2: type = converted %int.make_type_signed.loc36, %.loc36_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %w.var: ref %i32 = var w
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc36_30.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc36_30.2: <bound method> = bound_method %int.1, %.loc36_30.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc36_30.3: <specific function> = specific_function %.loc36_30.2, @Convert.2(constants.%int.32) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc36: init %i32 = call %.loc36_30.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc36_30.4: init %i32 = converted %int.1, %int.convert_checked.loc36 [template = constants.%int.1.2]
// CHECK:STDOUT:   assign %w.var, %.loc36_30.4
// CHECK:STDOUT:   br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc29:
// CHECK:STDOUT:   br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc27:
// CHECK:STDOUT:   %int.0.loc39: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc39_11.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc39_11.2: <bound method> = bound_method %int.0.loc39, %.loc39_11.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc39_11.3: <specific function> = specific_function %.loc39_11.2, @Convert.2(constants.%int.32) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc39: init %i32 = call %.loc39_11.3(%int.0.loc39) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc39_11.4: %i32 = value_of_initializer %int.convert_checked.loc39 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc39_11.5: %i32 = converted %int.0.loc39, %.loc39_11.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   return %.loc39_11.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
