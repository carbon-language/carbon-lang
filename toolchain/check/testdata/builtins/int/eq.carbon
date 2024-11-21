// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/eq.carbon

// --- int_eq.carbon

fn Eq(a: i32, b: i32) -> bool = "int.eq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Eq(1, 1) then True else False);
  false_ as (if Eq(1, 2) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Eq(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: error: invalid signature for builtin function "int.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";

// CHECK:STDOUT: --- int_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [template]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: bool = bool_literal true [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.38: bool = bool_literal false [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type = fn_decl @Eq [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_10: init type = call constants.%Int(%.loc2_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_10.2: type = value_of_initializer %int.make_type_signed.loc2_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_10.3: type = converted %int.make_type_signed.loc2_10, %.loc2_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%.loc2_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.3: type = converted %int.make_type_signed.loc2_18, %.loc2_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %bool.make_type, %.loc2_26.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {} {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %true_.patt: %True = binding_pattern true_
// CHECK:STDOUT:     %true_.param_patt: %True = value_param_pattern %true_.patt, runtime_param0
// CHECK:STDOUT:     %false_.patt: %False = binding_pattern false_
// CHECK:STDOUT:     %false_.param_patt: %False = value_param_pattern %false_.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %True.ref.loc7: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:     %False.ref.loc7: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:     %true_.param: %True = value_param runtime_param0
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param runtime_param1
// CHECK:STDOUT:     %false_: %False = bind_name false_, %false_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_19: init type = call constants.%Int(%.loc12_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_19.2: type = value_of_initializer %int.make_type_signed.loc12_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_19.3: type = converted %int.make_type_signed.loc12_19, %.loc12_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc12_27: init type = call constants.%Int(%.loc12_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_27.2: type = value_of_initializer %int.make_type_signed.loc12_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_27.3: type = converted %int.make_type_signed.loc12_27, %.loc12_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc12_35.2: type = converted %bool.make_type, %.loc12_35.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Eq(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Eq.ref.loc8: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %.loc8_19.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_22.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_19.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_19.3: <bound method> = bound_method %.loc8_19.1, %.loc8_19.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_19.4: <specific function> = specific_function %.loc8_19.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc8_19: init %i32 = call %.loc8_19.4(%.loc8_19.1) [template = constants.%.32]
// CHECK:STDOUT:   %.loc8_19.5: %i32 = value_of_initializer %int.convert_checked.loc8_19 [template = constants.%.32]
// CHECK:STDOUT:   %.loc8_19.6: %i32 = converted %.loc8_19.1, %.loc8_19.5 [template = constants.%.32]
// CHECK:STDOUT:   %.loc8_22.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_22.3: <bound method> = bound_method %.loc8_22.1, %.loc8_22.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_22.4: <specific function> = specific_function %.loc8_22.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc8_22: init %i32 = call %.loc8_22.4(%.loc8_22.1) [template = constants.%.32]
// CHECK:STDOUT:   %.loc8_22.5: %i32 = value_of_initializer %int.convert_checked.loc8_22 [template = constants.%.32]
// CHECK:STDOUT:   %.loc8_22.6: %i32 = converted %.loc8_22.1, %.loc8_22.5 [template = constants.%.32]
// CHECK:STDOUT:   %int.eq.loc8: init bool = call %Eq.ref.loc8(%.loc8_19.6, %.loc8_22.6) [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_13.1: bool = value_of_initializer %int.eq.loc8 [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_13.2: bool = converted %int.eq.loc8, %.loc8_13.1 [template = constants.%.33]
// CHECK:STDOUT:   if %.loc8_13.2 br !if.expr.then.loc8 else br !if.expr.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8:
// CHECK:STDOUT:   %True.ref.loc8: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc8(%True.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8:
// CHECK:STDOUT:   %False.ref.loc8: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc8(%False.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8:
// CHECK:STDOUT:   %.loc8_13.3: type = block_arg !if.expr.result.loc8 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Eq.ref.loc9: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %.loc9_20.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_23.1: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_20.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_20.3: <bound method> = bound_method %.loc9_20.1, %.loc9_20.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_20.4: <specific function> = specific_function %.loc9_20.3, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc9_20: init %i32 = call %.loc9_20.4(%.loc9_20.1) [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_20.5: %i32 = value_of_initializer %int.convert_checked.loc9_20 [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_20.6: %i32 = converted %.loc9_20.1, %.loc9_20.5 [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_23.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_23.3: <bound method> = bound_method %.loc9_23.1, %.loc9_23.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_23.4: <specific function> = specific_function %.loc9_23.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc9_23: init %i32 = call %.loc9_23.4(%.loc9_23.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_23.5: %i32 = value_of_initializer %int.convert_checked.loc9_23 [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_23.6: %i32 = converted %.loc9_23.1, %.loc9_23.5 [template = constants.%.37]
// CHECK:STDOUT:   %int.eq.loc9: init bool = call %Eq.ref.loc9(%.loc9_20.6, %.loc9_23.6) [template = constants.%.38]
// CHECK:STDOUT:   %.loc9_14.1: bool = value_of_initializer %int.eq.loc9 [template = constants.%.38]
// CHECK:STDOUT:   %.loc9_14.2: bool = converted %int.eq.loc9, %.loc9_14.1 [template = constants.%.38]
// CHECK:STDOUT:   if %.loc9_14.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %True.ref.loc9: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc9(%True.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %False.ref.loc9: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc9(%False.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_14.3: type = block_arg !if.expr.result.loc9 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.eq: init bool = call %Eq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc13_18.1: bool = value_of_initializer %int.eq
// CHECK:STDOUT:   %.loc13_18.2: bool = converted %int.eq, %.loc13_18.1
// CHECK:STDOUT:   return %.loc13_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %WrongResult.type: type = fn_type @WrongResult [template]
// CHECK:STDOUT:   %WrongResult: %WrongResult.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrongResult = %WrongResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongResult.decl: %WrongResult.type = fn_decl @WrongResult [template = constants.%WrongResult] {
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
// CHECK:STDOUT: fn @WrongResult(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
