// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_bound_negative.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_bound_negative.carbon

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bound_negative.carbon:[[@LINE+3]]:14: error: array bound of -1 is negative [ArrayBoundNegative]
// CHECK:STDERR: var a: [i32; Negate(1)];
// CHECK:STDERR:              ^~~~~~~~~
var a: [i32; Negate(1)];

// CHECK:STDOUT: --- fail_bound_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1.1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_-1.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: Core.IntLiteral = int_value -1 [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_14: init type = call constants.%Int(%int_32.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_signed.loc11_14, %.loc11_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_22: init type = call constants.%Int(%int_32.loc11_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_22.1: type = value_of_initializer %int.make_type_signed.loc11_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_22.2: type = converted %int.make_type_signed.loc11_22, %.loc11_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_21: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_21: <bound method> = bound_method %int_1, %impl.elem0.loc16_21 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_21: <specific function> = specific_function %Convert.bound.loc16_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_21: init %i32 = call %Convert.specific_fn.loc16_21(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_21.1: %i32 = value_of_initializer %int.convert_checked.loc16_21 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_21.2: %i32 = converted %int_1, %.loc16_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc16_21.2) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc16_9.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_9.2: type = converted %int.make_type_signed, %.loc16_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_22: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc16_22: <bound method> = bound_method %int.snegate, %impl.elem0.loc16_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_22: <specific function> = specific_function %Convert.bound.loc16_22, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc16_22.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc16_22.2: %i32 = converted %int.snegate, %.loc16_22.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_22: init Core.IntLiteral = call %Convert.specific_fn.loc16_22(%.loc16_22.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_22.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc16_22 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc16_22.4: Core.IntLiteral = converted %int.snegate, %.loc16_22.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %array_type: type = array_type %.loc16_22.4, %i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
