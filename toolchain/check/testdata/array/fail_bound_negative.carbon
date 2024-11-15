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
// CHECK:STDERR:              ^~~~~~~
var a: [i32; Negate(1)];

// CHECK:STDOUT: --- fail_bound_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.28: i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.28, %Convert.16 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value -1 [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_32.loc11_14, %.loc11_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_22.1: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_22.2: type = converted %int.make_type_32.loc11_22, %.loc11_22.1 [template = i32]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc16_21.1: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_21.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_21.3: <bound method> = bound_method %.loc16_21.1, %.loc16_21.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc16_21: init i32 = call %.loc16_21.3(%.loc16_21.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc16_21.4: i32 = value_of_initializer %int.convert_checked.loc16_21 [template = constants.%.27]
// CHECK:STDOUT:   %.loc16_21.5: i32 = converted %.loc16_21.1, %.loc16_21.4 [template = constants.%.27]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc16_21.5) [template = constants.%.28]
// CHECK:STDOUT:   %.loc16_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_9.2: type = converted %int.make_type_32, %.loc16_9.1 [template = i32]
// CHECK:STDOUT:   %.loc16_20.1: %Convert.type.5 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc16_20.2: <bound method> = bound_method %int.snegate, %.loc16_20.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_20.3: i32 = value_of_initializer %int.snegate [template = constants.%.28]
// CHECK:STDOUT:   %.loc16_20.4: i32 = converted %int.snegate, %.loc16_20.3 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc16_20: init Core.IntLiteral = call %.loc16_20.2(%.loc16_20.4) [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_20.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc16_20 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_20.6: Core.IntLiteral = converted %int.snegate, %.loc16_20.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_23: type = array_type %.loc16_20.6, i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
