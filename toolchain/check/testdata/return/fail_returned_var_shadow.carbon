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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %SameScope.type: type = fn_type @SameScope [template]
// CHECK:STDOUT:   %SameScope: %SameScope.type = struct_value () [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
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
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_signed.loc11, %.loc11_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentScopes.decl: %DifferentScopes.type = fn_decl @DifferentScopes [template = constants.%DifferentScopes] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc26: init type = call constants.%Int(%int_32.loc26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_25.1: type = value_of_initializer %int.make_type_signed.loc26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_25.2: type = converted %int.make_type_signed.loc26, %.loc26_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_signed.loc13, %.loc13_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc13: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_28: init %i32 = converted %int_0.loc13, %int.convert_checked.loc13 [template = constants.%int_0.2]
// CHECK:STDOUT:   assign %v.var, %.loc13_28
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int_32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_21.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_21.2: type = converted %int.make_type_signed.loc21, %.loc21_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %w.var: ref %i32 = var w
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_1, %impl.elem0.loc21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %Convert.specific_fn.loc21(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc21_28: init %i32 = converted %int_1, %int.convert_checked.loc21 [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %w.var, %.loc21_28
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc23: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc23: <bound method> = bound_method %int_0.loc23, %impl.elem0.loc23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc23: <specific function> = specific_function %Convert.bound.loc23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %Convert.specific_fn.loc23(%int_0.loc23) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc23_11.1: %i32 = value_of_initializer %int.convert_checked.loc23 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc23_11.2: %i32 = converted %int_0.loc23, %.loc23_11.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   return %.loc23_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc27: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc27 br !if.then.loc27 else br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc27:
// CHECK:STDOUT:   %int_32.loc28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc28: init type = call constants.%Int(%int_32.loc28) [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_21.1: type = value_of_initializer %int.make_type_signed.loc28 [template = constants.%i32]
// CHECK:STDOUT:   %.loc28_21.2: type = converted %int.make_type_signed.loc28, %.loc28_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %v.var: ref %i32 = var v
// CHECK:STDOUT:   %v: ref %i32 = bind_name v, %v.var
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc28: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc28: <bound method> = bound_method %int_0.loc28, %impl.elem0.loc28 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc28: <specific function> = specific_function %Convert.bound.loc28, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc28: init %i32 = call %Convert.specific_fn.loc28(%int_0.loc28) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc28_28: init %i32 = converted %int_0.loc28, %int.convert_checked.loc28 [template = constants.%int_0.2]
// CHECK:STDOUT:   assign %v.var, %.loc28_28
// CHECK:STDOUT:   %true.loc29: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc29 br !if.then.loc29 else br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc29:
// CHECK:STDOUT:   %int_32.loc36: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc36: init type = call constants.%Int(%int_32.loc36) [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_23.1: type = value_of_initializer %int.make_type_signed.loc36 [template = constants.%i32]
// CHECK:STDOUT:   %.loc36_23.2: type = converted %int.make_type_signed.loc36, %.loc36_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %w.var: ref %i32 = var w
// CHECK:STDOUT:   %w: ref %i32 = bind_name w, %w.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc36: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc36: <bound method> = bound_method %int_1, %impl.elem0.loc36 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc36: <specific function> = specific_function %Convert.bound.loc36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc36: init %i32 = call %Convert.specific_fn.loc36(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc36_30: init %i32 = converted %int_1, %int.convert_checked.loc36 [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %w.var, %.loc36_30
// CHECK:STDOUT:   br !if.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc29:
// CHECK:STDOUT:   br !if.else.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc27:
// CHECK:STDOUT:   %int_0.loc39: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc39: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc39: <bound method> = bound_method %int_0.loc39, %impl.elem0.loc39 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc39: <specific function> = specific_function %Convert.bound.loc39, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc39: init %i32 = call %Convert.specific_fn.loc39(%int_0.loc39) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc39_11.1: %i32 = value_of_initializer %int.convert_checked.loc39 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc39_11.2: %i32 = converted %int_0.loc39, %.loc39_11.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   return %.loc39_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
