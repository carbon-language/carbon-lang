// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/call.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

var arr: [i32; Add(1, 2)];

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.28: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.31, %Convert.16 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.35: type = array_type %.34, i32 [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc13_20.1: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_23.1: Core.IntLiteral = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_20.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_20.3: <bound method> = bound_method %.loc13_20.1, %.loc13_20.2 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc13_20: init i32 = call %.loc13_20.3(%.loc13_20.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_20.4: i32 = value_of_initializer %int.convert_checked.loc13_20 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_20.5: i32 = converted %.loc13_20.1, %.loc13_20.4 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_23.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_23.3: <bound method> = bound_method %.loc13_23.1, %.loc13_23.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc13_23: init i32 = call %.loc13_23.3(%.loc13_23.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_23.4: i32 = value_of_initializer %int.convert_checked.loc13_23 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_23.5: i32 = converted %.loc13_23.1, %.loc13_23.4 [template = constants.%.30]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc13_20.5, %.loc13_23.5) [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc13_11.2: type = converted %int.make_type_32, %.loc13_11.1 [template = i32]
// CHECK:STDOUT:   %.loc13_19.1: %Convert.type.5 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc13_19.2: <bound method> = bound_method %int.sadd, %.loc13_19.1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_19.3: i32 = value_of_initializer %int.sadd [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_19.4: i32 = converted %int.sadd, %.loc13_19.3 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc13_19: init Core.IntLiteral = call %.loc13_19.2(%.loc13_19.4) [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_19.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_19 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_19.6: Core.IntLiteral = converted %int.sadd, %.loc13_19.5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_25: type = array_type %.loc13_19.6, i32 [template = constants.%.35]
// CHECK:STDOUT:   %arr.var: ref %.35 = var arr
// CHECK:STDOUT:   %arr: ref %.35 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
