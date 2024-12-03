// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/and.carbon

// --- int_and.carbon

fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(12, 10)];
let arr_p: [i32; 8]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return And(a, b);
}

// CHECK:STDOUT: --- int_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int.10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.25: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %.26: <specific function> = specific_function %.25, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.10.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.10.2: %i32 = int_value 10 [template]
// CHECK:STDOUT:   %int.8.1: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int.32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %int.8.1, %Convert.15 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.4(%int.32) [template]
// CHECK:STDOUT:   %int.8.2: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %.32: type = array_type %int.8.2, %i32 [template]
// CHECK:STDOUT:   %.33: type = ptr_type %.32 [template]
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
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%int.32.loc2_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_signed.loc2_11, %.loc2_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%int.32.loc2_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_signed.loc2_19, %.loc2_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%int.32.loc2_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_signed.loc2_27, %.loc2_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%int.32.loc4) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %int.12: Core.IntLiteral = int_value 12 [template = constants.%int.12.1]
// CHECK:STDOUT:   %int.10: Core.IntLiteral = int_value 10 [template = constants.%int.10.1]
// CHECK:STDOUT:   %.loc4_20.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_20.2: <bound method> = bound_method %int.12, %.loc4_20.1 [template = constants.%.25]
// CHECK:STDOUT:   %.loc4_20.3: <specific function> = specific_function %.loc4_20.2, @Convert.2(constants.%int.32) [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init %i32 = call %.loc4_20.3(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc4_20.4: %i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc4_20.5: %i32 = converted %int.12, %.loc4_20.4 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc4_24.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_24.2: <bound method> = bound_method %int.10, %.loc4_24.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_24.3: <specific function> = specific_function %.loc4_24.2, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc4_24: init %i32 = call %.loc4_24.3(%int.10) [template = constants.%int.10.2]
// CHECK:STDOUT:   %.loc4_24.4: %i32 = value_of_initializer %int.convert_checked.loc4_24 [template = constants.%int.10.2]
// CHECK:STDOUT:   %.loc4_24.5: %i32 = converted %int.10, %.loc4_24.4 [template = constants.%int.10.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc4_20.5, %.loc4_24.5) [template = constants.%int.8.1]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_26.1: %Convert.type.6 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_26.2: <bound method> = bound_method %int.and, %.loc4_26.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_26.3: <specific function> = specific_function %.loc4_26.2, @Convert.4(constants.%int.32) [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_26.4: %i32 = value_of_initializer %int.and [template = constants.%int.8.1]
// CHECK:STDOUT:   %.loc4_26.5: %i32 = converted %int.and, %.loc4_26.4 [template = constants.%int.8.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_26: init Core.IntLiteral = call %.loc4_26.3(%.loc4_26.5) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc4_26.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_26 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc4_26.7: Core.IntLiteral = converted %int.and, %.loc4_26.6 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc4_27: type = array_type %.loc4_26.7, %i32 [template = constants.%.32]
// CHECK:STDOUT:   %arr.var: ref %.32 = var arr
// CHECK:STDOUT:   %arr: ref %.32 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %int.8: Core.IntLiteral = int_value 8 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_signed.loc5, %.loc5_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %int.8, %i32 [template = constants.%.32]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.32 [template = constants.%.33]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%int.32.loc7_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_signed.loc7_19, %.loc7_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%int.32.loc7_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_signed.loc7_27, %.loc7_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_35: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%int.32.loc7_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_signed.loc7_35, %.loc7_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.and, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.32 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.33 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.33 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
