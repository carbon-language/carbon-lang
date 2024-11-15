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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.30: type = array_type %.29, i32 [template]
// CHECK:STDOUT:   %.31: type = ptr_type %.30 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
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
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc2_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_32.loc2_11 [template = i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_32.loc2_11, %.loc2_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_32.loc2_19 [template = i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_32.loc2_19, %.loc2_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_32.loc2_27 [template = i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_32.loc2_27, %.loc2_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, %Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc4_20: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %int.usub: init i32 = call %Sub.ref(%.loc4_20, %.loc4_23) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %int.usub, %.loc4_19.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_19.3: i32 = value_of_initializer %int.usub [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_19.4: i32 = converted %int.usub, %.loc4_19.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc4: init Core.IntLiteral = call %.loc4_19.2(%.loc4_19.4) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_19.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_19.6: Core.IntLiteral = converted %int.usub, %.loc4_19.5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19.6, i32 [template = constants.%.30]
// CHECK:STDOUT:   %arr.var: ref %.30 = var arr
// CHECK:STDOUT:   %arr: ref %.30 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18.1: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc5_18.3: <bound method> = bound_method %.loc5_18.1, %.loc5_18.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc5: init Core.IntLiteral = call %.loc5_18.3(%.loc5_18.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_18.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_18.5: Core.IntLiteral = converted %.loc5_18.1, %.loc5_18.4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18.5, i32 [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.30 [template = constants.%.31]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_32.loc7_27 [template = i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_32.loc7_27, %.loc7_27.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_32.loc7_35 [template = i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_32.loc7_35, %.loc7_35.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: i32, %b.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.usub: init i32 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: i32 = value_of_initializer %int.usub
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %int.usub, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.30 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.31 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.31 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %.6: i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
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
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc4_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4_11 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4_11, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_32.loc4_19 [template = i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_32.loc4_19, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4_27 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4_27, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32.loc8, %.loc8_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc6: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc6_18: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_21: i32 = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.usub.loc6: init i32 = call %Sub.ref.loc6(%.loc6_18, %.loc6_21) [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_32.1: i32 = value_of_initializer %int.usub.loc6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_32.2: i32 = converted %int.usub.loc6, %.loc6_32.1 [template = constants.%.3]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Sub.ref.loc7_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc7_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc7_22: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_25: i32 = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.usub.loc7_21: init i32 = call %Sub.ref.loc7_18(%.loc7_22, %.loc7_25) [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_38: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_21.1: i32 = value_of_initializer %int.usub.loc7_21 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_21.2: i32 = converted %int.usub.loc7_21, %.loc7_21.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.usub.loc7_17: init i32 = call %Sub.ref.loc7_14(%.loc7_21.2, %.loc7_38) [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_40.1: i32 = value_of_initializer %int.usub.loc7_17 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_40.2: i32 = converted %int.usub.loc7_17, %.loc7_40.1 [template = constants.%.5]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc7_40.2
// CHECK:STDOUT:   %Sub.ref.loc8_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc8_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %.loc8_22: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_25: i32 = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.usub.loc8_21: init i32 = call %Sub.ref.loc8_18(%.loc8_22, %.loc8_25) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_38: i32 = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_21.1: i32 = value_of_initializer %int.usub.loc8_21 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_21.2: i32 = converted %int.usub.loc8_21, %.loc8_21.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.usub.loc8_17: init i32 = call %Sub.ref.loc8_14(%.loc8_21.2, %.loc8_38) [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.1: i32 = value_of_initializer %int.usub.loc8_17 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.2: i32 = converted %int.usub.loc8_17, %.loc8_40.1 [template = constants.%.2]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc8_40.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
