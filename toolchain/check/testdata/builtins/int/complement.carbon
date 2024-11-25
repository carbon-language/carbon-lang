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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1193046 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1193046 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value -1193047 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 16777215 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 16777215 [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 15584169 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.36: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.35, %Convert.15 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 15584169 [template]
// CHECK:STDOUT:   %.40: type = array_type %.39, %i32 [template]
// CHECK:STDOUT:   %.41: type = ptr_type %.40 [template]
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
// CHECK:STDOUT:     %.loc2_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%.loc2_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.3: type = converted %int.make_type_signed.loc2_18, %.loc2_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_26: init type = call constants.%Int(%.loc2_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.2: type = value_of_initializer %int.make_type_signed.loc2_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.3: type = converted %int.make_type_signed.loc2_26, %.loc2_26.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %.loc3_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc3_11: init type = call constants.%Int(%.loc3_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.2: type = value_of_initializer %int.make_type_signed.loc3_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.3: type = converted %int.make_type_signed.loc3_11, %.loc3_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc3_19: init type = call constants.%Int(%.loc3_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.2: type = value_of_initializer %int.make_type_signed.loc3_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.3: type = converted %int.make_type_signed.loc3_19, %.loc3_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc3_27: init type = call constants.%Int(%.loc3_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.2: type = value_of_initializer %int.make_type_signed.loc3_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.3: type = converted %int.make_type_signed.loc3_27, %.loc3_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, %Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %.loc5_31.1: Core.IntLiteral = int_value 1193046 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_31.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_31.3: <bound method> = bound_method %.loc5_31.1, %.loc5_31.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc5_31.4: <specific function> = specific_function %.loc5_31.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc5_31: init %i32 = call %.loc5_31.4(%.loc5_31.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_31.5: %i32 = value_of_initializer %int.convert_checked.loc5_31 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_31.6: %i32 = converted %.loc5_31.1, %.loc5_31.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%.loc5_31.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_42.1: Core.IntLiteral = int_value 16777215 [template = constants.%.31]
// CHECK:STDOUT:   %.loc5_30.1: %i32 = value_of_initializer %int.complement [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_30.2: %i32 = converted %int.complement, %.loc5_30.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_42.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc5_42.3: <bound method> = bound_method %.loc5_42.1, %.loc5_42.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc5_42.4: <specific function> = specific_function %.loc5_42.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc5_42: init %i32 = call %.loc5_42.4(%.loc5_42.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc5_42.5: %i32 = value_of_initializer %int.convert_checked.loc5_42 [template = constants.%.34]
// CHECK:STDOUT:   %.loc5_42.6: %i32 = converted %.loc5_42.1, %.loc5_42.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc5_30.2, %.loc5_42.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc5_11.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_11.3: type = converted %int.make_type_signed.loc5, %.loc5_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19.1: %Convert.type.6 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc5_19.2: <bound method> = bound_method %int.and, %.loc5_19.1 [template = constants.%.37]
// CHECK:STDOUT:   %.loc5_19.3: <specific function> = specific_function %.loc5_19.2, @Convert.4(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_19.4: %i32 = value_of_initializer %int.and [template = constants.%.35]
// CHECK:STDOUT:   %.loc5_19.5: %i32 = converted %int.and, %.loc5_19.4 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc5_19: init Core.IntLiteral = call %.loc5_19.3(%.loc5_19.5) [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_19.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5_19 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_19.7: Core.IntLiteral = converted %int.and, %.loc5_19.6 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_51: type = array_type %.loc5_19.7, %i32 [template = constants.%.40]
// CHECK:STDOUT:   %arr.var: ref %.40 = var arr
// CHECK:STDOUT:   %arr: ref %.40 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc6_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_18: Core.IntLiteral = int_value 15584169 [template = constants.%.39]
// CHECK:STDOUT:   %.loc6_13.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_13.3: type = converted %int.make_type_signed.loc6, %.loc6_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_26: type = array_type %.loc6_18, %i32 [template = constants.%.40]
// CHECK:STDOUT:   %.loc6_27: type = ptr_type %.40 [template = constants.%.41]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc8_19: init type = call constants.%Int(%.loc8_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_19.2: type = value_of_initializer %int.make_type_signed.loc8_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_19.3: type = converted %int.make_type_signed.loc8_19, %.loc8_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc8_27: init type = call constants.%Int(%.loc8_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.2: type = value_of_initializer %int.make_type_signed.loc8_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.3: type = converted %int.make_type_signed.loc8_27, %.loc8_27.2 [template = constants.%i32]
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
// CHECK:STDOUT:   %arr.ref: ref %.40 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc6: %.41 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.41 = bind_name arr_p, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
