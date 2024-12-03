// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/complement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/complement.carbon

// --- int_complement.carbon

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
let arr_p: [i32; 0xEDCBA9]* = &arr;

fn RuntimeCall(a: i32) -> i32 {
  return Complement(a);
}

// CHECK:STDOUT: --- int_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int.1193046.1: Core.IntLiteral = int_value 1193046 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.25: <bound method> = bound_method %int.1193046.1, %Convert.14 [template]
// CHECK:STDOUT:   %.26: <specific function> = specific_function %.25, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1193046.2: %i32 = int_value 1193046 [template]
// CHECK:STDOUT:   %int.-1193047: %i32 = int_value -1193047 [template]
// CHECK:STDOUT:   %int.16777215.1: Core.IntLiteral = int_value 16777215 [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.16777215.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.16777215.2: %i32 = int_value 16777215 [template]
// CHECK:STDOUT:   %int.15584169.1: %i32 = int_value 15584169 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int.32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %int.15584169.1, %Convert.15 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.4(%int.32) [template]
// CHECK:STDOUT:   %int.15584169.2: Core.IntLiteral = int_value 15584169 [template]
// CHECK:STDOUT:   %.32: type = array_type %int.15584169.2, %i32 [template]
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
// CHECK:STDOUT:     .Complement = %Complement.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Complement.decl: %Complement.type = fn_decl @Complement [template = constants.%Complement] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc2_18: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%int.32.loc2_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_signed.loc2_18, %.loc2_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_26: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_26: init type = call constants.%Int(%int.32.loc2_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_signed.loc2_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_signed.loc2_26, %.loc2_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc3_11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_11: init type = call constants.%Int(%int.32.loc3_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.1: type = value_of_initializer %int.make_type_signed.loc3_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.2: type = converted %int.make_type_signed.loc3_11, %.loc3_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc3_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_19: init type = call constants.%Int(%int.32.loc3_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.1: type = value_of_initializer %int.make_type_signed.loc3_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.2: type = converted %int.make_type_signed.loc3_19, %.loc3_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc3_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_27: init type = call constants.%Int(%int.32.loc3_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.1: type = value_of_initializer %int.make_type_signed.loc3_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.2: type = converted %int.make_type_signed.loc3_27, %.loc3_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, %Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int.1193046: Core.IntLiteral = int_value 1193046 [template = constants.%int.1193046.1]
// CHECK:STDOUT:   %.loc5_31.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_31.2: <bound method> = bound_method %int.1193046, %.loc5_31.1 [template = constants.%.25]
// CHECK:STDOUT:   %.loc5_31.3: <specific function> = specific_function %.loc5_31.2, @Convert.2(constants.%int.32) [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc5_31: init %i32 = call %.loc5_31.3(%int.1193046) [template = constants.%int.1193046.2]
// CHECK:STDOUT:   %.loc5_31.4: %i32 = value_of_initializer %int.convert_checked.loc5_31 [template = constants.%int.1193046.2]
// CHECK:STDOUT:   %.loc5_31.5: %i32 = converted %int.1193046, %.loc5_31.4 [template = constants.%int.1193046.2]
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%.loc5_31.5) [template = constants.%int.-1193047]
// CHECK:STDOUT:   %int.16777215: Core.IntLiteral = int_value 16777215 [template = constants.%int.16777215.1]
// CHECK:STDOUT:   %.loc5_39.1: %i32 = value_of_initializer %int.complement [template = constants.%int.-1193047]
// CHECK:STDOUT:   %.loc5_39.2: %i32 = converted %int.complement, %.loc5_39.1 [template = constants.%int.-1193047]
// CHECK:STDOUT:   %.loc5_42.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_42.2: <bound method> = bound_method %int.16777215, %.loc5_42.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc5_42.3: <specific function> = specific_function %.loc5_42.2, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc5_42: init %i32 = call %.loc5_42.3(%int.16777215) [template = constants.%int.16777215.2]
// CHECK:STDOUT:   %.loc5_42.4: %i32 = value_of_initializer %int.convert_checked.loc5_42 [template = constants.%int.16777215.2]
// CHECK:STDOUT:   %.loc5_42.5: %i32 = converted %int.16777215, %.loc5_42.4 [template = constants.%int.16777215.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc5_39.2, %.loc5_42.5) [template = constants.%int.15584169.1]
// CHECK:STDOUT:   %.loc5_11.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_11.2: type = converted %int.make_type_signed.loc5, %.loc5_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_50.1: %Convert.type.6 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc5_50.2: <bound method> = bound_method %int.and, %.loc5_50.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_50.3: <specific function> = specific_function %.loc5_50.2, @Convert.4(constants.%int.32) [template = constants.%.31]
// CHECK:STDOUT:   %.loc5_50.4: %i32 = value_of_initializer %int.and [template = constants.%int.15584169.1]
// CHECK:STDOUT:   %.loc5_50.5: %i32 = converted %int.and, %.loc5_50.4 [template = constants.%int.15584169.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_50: init Core.IntLiteral = call %.loc5_50.3(%.loc5_50.5) [template = constants.%int.15584169.2]
// CHECK:STDOUT:   %.loc5_50.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5_50 [template = constants.%int.15584169.2]
// CHECK:STDOUT:   %.loc5_50.7: Core.IntLiteral = converted %int.and, %.loc5_50.6 [template = constants.%int.15584169.2]
// CHECK:STDOUT:   %.loc5_51: type = array_type %.loc5_50.7, %i32 [template = constants.%.32]
// CHECK:STDOUT:   %arr.var: ref %.32 = var arr
// CHECK:STDOUT:   %arr: ref %.32 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int.32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %int.15584169: Core.IntLiteral = int_value 15584169 [template = constants.%int.15584169.2]
// CHECK:STDOUT:   %.loc6_13.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_13.2: type = converted %int.make_type_signed.loc6, %.loc6_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_26: type = array_type %int.15584169, %i32 [template = constants.%.32]
// CHECK:STDOUT:   %.loc6_27: type = ptr_type %.32 [template = constants.%.33]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc8_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_19: init type = call constants.%Int(%int.32.loc8_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_19.1: type = value_of_initializer %int.make_type_signed.loc8_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_19.2: type = converted %int.make_type_signed.loc8_19, %.loc8_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc8_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_27: init type = call constants.%Int(%int.32.loc8_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.1: type = value_of_initializer %int.make_type_signed.loc8_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.2: type = converted %int.make_type_signed.loc8_27, %.loc8_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Complement(%a.param_patt: %i32) -> %i32 = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%a.ref)
// CHECK:STDOUT:   %.loc9_23.1: %i32 = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc9_23.2: %i32 = converted %int.complement, %.loc9_23.1
// CHECK:STDOUT:   return %.loc9_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.32 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc6: %.33 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.33 = bind_name arr_p, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
