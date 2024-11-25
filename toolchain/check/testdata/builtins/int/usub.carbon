// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/usub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/usub.carbon

// --- int_sub.carbon

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

var arr: [i32; Sub(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Sub(a, b);
}

// --- overflow.carbon

package Overflow;

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

let a: i32 = Sub(0, 0x7FFFFFFF);
let b: i32 = Sub(Sub(0, 0x7FFFFFFF), 1);
let c: i32 = Sub(Sub(0, 0x7FFFFFFF), 2);

// CHECK:STDOUT: --- int_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.39: type = array_type %.38, %i32 [template]
// CHECK:STDOUT:   %.40: type = ptr_type %.39 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
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
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%.loc2_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.3: type = converted %int.make_type_signed.loc2_11, %.loc2_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%.loc2_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.3: type = converted %int.make_type_signed.loc2_19, %.loc2_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%.loc2_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.3: type = converted %int.make_type_signed.loc2_27, %.loc2_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, %Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc4_20.1: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23.1: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_20.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_20.3: <bound method> = bound_method %.loc4_20.1, %.loc4_20.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_20.4: <specific function> = specific_function %.loc4_20.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init %i32 = call %.loc4_20.4(%.loc4_20.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.5: %i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.6: %i32 = converted %.loc4_20.1, %.loc4_20.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_23.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_23.3: <bound method> = bound_method %.loc4_23.1, %.loc4_23.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_23.4: <specific function> = specific_function %.loc4_23.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc4_23: init %i32 = call %.loc4_23.4(%.loc4_23.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.5: %i32 = value_of_initializer %int.convert_checked.loc4_23 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.6: %i32 = converted %.loc4_23.1, %.loc4_23.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%.loc4_20.6, %.loc4_23.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.6 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %int.usub, %.loc4_19.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_19.3: <specific function> = specific_function %.loc4_19.2, @Convert.4(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc4_19.4: %i32 = value_of_initializer %int.usub [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_19.5: %i32 = converted %int.usub, %.loc4_19.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc4_19: init Core.IntLiteral = call %.loc4_19.3(%.loc4_19.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_19 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.7: Core.IntLiteral = converted %int.usub, %.loc4_19.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19.7, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %arr.var: ref %.39 = var arr
// CHECK:STDOUT:   %arr: ref %.39 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_13.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.3: type = converted %int.make_type_signed.loc5, %.loc5_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.39 [template = constants.%.40]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%.loc7_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %int.make_type_signed.loc7_19, %.loc7_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%.loc7_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %int.make_type_signed.loc7_27, %.loc7_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%.loc7_35.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.3: type = converted %int.make_type_signed.loc7_35, %.loc7_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.usub
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.usub, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.39 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.40 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.40 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.35, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.39: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.40, %Convert.14 [template]
// CHECK:STDOUT:   %.42: <specific function> = specific_function %.41, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.43: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%.loc4_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %int.make_type_signed.loc4_19, %.loc4_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed.loc4_27, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.3: type = converted %int.make_type_signed.loc6, %.loc6_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.3: type = converted %int.make_type_signed.loc7, %.loc7_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.3: type = converted %int.make_type_signed.loc8, %.loc8_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc6: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc6_18.1: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_21.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_18.3: <bound method> = bound_method %.loc6_18.1, %.loc6_18.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc6_18.4: <specific function> = specific_function %.loc6_18.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init %i32 = call %.loc6_18.4(%.loc6_18.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_18.5: %i32 = value_of_initializer %int.convert_checked.loc6_18 [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_18.6: %i32 = converted %.loc6_18.1, %.loc6_18.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_21.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_21.3: <bound method> = bound_method %.loc6_21.1, %.loc6_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_21.4: <specific function> = specific_function %.loc6_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc6_21: init %i32 = call %.loc6_21.4(%.loc6_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc6_21.5: %i32 = value_of_initializer %int.convert_checked.loc6_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc6_21.6: %i32 = converted %.loc6_21.1, %.loc6_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.usub.loc6: init %i32 = call %Sub.ref.loc6(%.loc6_18.6, %.loc6_21.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %int.usub.loc6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %int.usub.loc6, %.loc6_32.1 [template = constants.%.34]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Sub.ref.loc7_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc7_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc7_22.1: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_25.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_22.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_22.3: <bound method> = bound_method %.loc7_22.1, %.loc7_22.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc7_22.4: <specific function> = specific_function %.loc7_22.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc7_22: init %i32 = call %.loc7_22.4(%.loc7_22.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc7_22.5: %i32 = value_of_initializer %int.convert_checked.loc7_22 [template = constants.%.30]
// CHECK:STDOUT:   %.loc7_22.6: %i32 = converted %.loc7_22.1, %.loc7_22.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc7_25.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_25.3: <bound method> = bound_method %.loc7_25.1, %.loc7_25.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc7_25.4: <specific function> = specific_function %.loc7_25.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc7_25: init %i32 = call %.loc7_25.4(%.loc7_25.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc7_25.5: %i32 = value_of_initializer %int.convert_checked.loc7_25 [template = constants.%.33]
// CHECK:STDOUT:   %.loc7_25.6: %i32 = converted %.loc7_25.1, %.loc7_25.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.usub.loc7_21: init %i32 = call %Sub.ref.loc7_18(%.loc7_22.6, %.loc7_25.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_38.1: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc7_21.1: %i32 = value_of_initializer %int.usub.loc7_21 [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_21.2: %i32 = converted %int.usub.loc7_21, %.loc7_21.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_38.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_38.3: <bound method> = bound_method %.loc7_38.1, %.loc7_38.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc7_38.4: <specific function> = specific_function %.loc7_38.3, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc7_38: init %i32 = call %.loc7_38.4(%.loc7_38.1) [template = constants.%.38]
// CHECK:STDOUT:   %.loc7_38.5: %i32 = value_of_initializer %int.convert_checked.loc7_38 [template = constants.%.38]
// CHECK:STDOUT:   %.loc7_38.6: %i32 = converted %.loc7_38.1, %.loc7_38.5 [template = constants.%.38]
// CHECK:STDOUT:   %int.usub.loc7_17: init %i32 = call %Sub.ref.loc7_14(%.loc7_21.2, %.loc7_38.6) [template = constants.%.39]
// CHECK:STDOUT:   %.loc7_40.1: %i32 = value_of_initializer %int.usub.loc7_17 [template = constants.%.39]
// CHECK:STDOUT:   %.loc7_40.2: %i32 = converted %int.usub.loc7_17, %.loc7_40.1 [template = constants.%.39]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc7_40.2
// CHECK:STDOUT:   %Sub.ref.loc8_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc8_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc8_22.1: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_22.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_22.3: <bound method> = bound_method %.loc8_22.1, %.loc8_22.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc8_22.4: <specific function> = specific_function %.loc8_22.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc8_22: init %i32 = call %.loc8_22.4(%.loc8_22.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_22.5: %i32 = value_of_initializer %int.convert_checked.loc8_22 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_22.6: %i32 = converted %.loc8_22.1, %.loc8_22.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_25.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_25.3: <bound method> = bound_method %.loc8_25.1, %.loc8_25.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc8_25.4: <specific function> = specific_function %.loc8_25.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc8_25: init %i32 = call %.loc8_25.4(%.loc8_25.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_25.5: %i32 = value_of_initializer %int.convert_checked.loc8_25 [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_25.6: %i32 = converted %.loc8_25.1, %.loc8_25.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.usub.loc8_21: init %i32 = call %Sub.ref.loc8_18(%.loc8_22.6, %.loc8_25.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_38.1: Core.IntLiteral = int_value 2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc8_21.1: %i32 = value_of_initializer %int.usub.loc8_21 [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_21.2: %i32 = converted %int.usub.loc8_21, %.loc8_21.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_38.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_38.3: <bound method> = bound_method %.loc8_38.1, %.loc8_38.2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc8_38.4: <specific function> = specific_function %.loc8_38.3, @Convert.2(constants.%.1) [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc8_38: init %i32 = call %.loc8_38.4(%.loc8_38.1) [template = constants.%.43]
// CHECK:STDOUT:   %.loc8_38.5: %i32 = value_of_initializer %int.convert_checked.loc8_38 [template = constants.%.43]
// CHECK:STDOUT:   %.loc8_38.6: %i32 = converted %.loc8_38.1, %.loc8_38.5 [template = constants.%.43]
// CHECK:STDOUT:   %int.usub.loc8_17: init %i32 = call %Sub.ref.loc8_14(%.loc8_21.2, %.loc8_38.6) [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_40.1: %i32 = value_of_initializer %int.usub.loc8_17 [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_40.2: %i32 = converted %int.usub.loc8_17, %.loc8_40.1 [template = constants.%.33]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc8_40.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
