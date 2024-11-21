// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/less.carbon

// --- int_less.carbon

fn Less(a: i32, b: i32) -> bool = "int.less";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1, 2) then True else False);
  false_ as (if Less(1, 1) then True else False);
  false_ as (if Less(1, 0) then True else False);
  true_ as (if Less(Negate(1), 0) then True else False);
  false_ as (if Less(0, Negate(1)) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Less(a, b);
}

// CHECK:STDOUT: --- int_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [template]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.37: bool = bool_literal true [template]
// CHECK:STDOUT:   %.38: bool = bool_literal false [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.14 [template]
// CHECK:STDOUT:   %.41: <specific function> = specific_function %.40, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.42: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.43: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .Less = %Less.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Less.decl: %Less.type = fn_decl @Less [template = constants.%Less] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_12: init type = call constants.%Int(%.loc2_12.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_12.2: type = value_of_initializer %int.make_type_signed.loc2_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_12.3: type = converted %int.make_type_signed.loc2_12, %.loc2_12.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_20: init type = call constants.%Int(%.loc2_20.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_20.2: type = value_of_initializer %int.make_type_signed.loc2_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_20.3: type = converted %int.make_type_signed.loc2_20, %.loc2_20.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_28.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc2_28.2: type = converted %bool.make_type, %.loc2_28.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc3_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc3_14: init type = call constants.%Int(%.loc3_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_14.2: type = value_of_initializer %int.make_type_signed.loc3_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_14.3: type = converted %int.make_type_signed.loc3_14, %.loc3_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc3_22: init type = call constants.%Int(%.loc3_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_22.2: type = value_of_initializer %int.make_type_signed.loc3_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_22.3: type = converted %int.make_type_signed.loc3_22, %.loc3_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {} {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %true_.patt: %True = binding_pattern true_
// CHECK:STDOUT:     %true_.param_patt: %True = value_param_pattern %true_.patt, runtime_param0
// CHECK:STDOUT:     %false_.patt: %False = binding_pattern false_
// CHECK:STDOUT:     %false_.param_patt: %False = value_param_pattern %false_.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %True.ref.loc8: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:     %False.ref.loc8: type = name_ref False, file.%False.decl [template = constants.%False]
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
// CHECK:STDOUT:     %.loc16_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc16_19: init type = call constants.%Int(%.loc16_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.2: type = value_of_initializer %int.make_type_signed.loc16_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.3: type = converted %int.make_type_signed.loc16_19, %.loc16_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc16_27: init type = call constants.%Int(%.loc16_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.2: type = value_of_initializer %int.make_type_signed.loc16_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.3: type = converted %int.make_type_signed.loc16_27, %.loc16_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc16_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc16_35.2: type = converted %bool.make_type, %.loc16_35.1 [template = bool]
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
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.less";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc9: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc9_21.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_24.1: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_21.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_21.3: <bound method> = bound_method %.loc9_21.1, %.loc9_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc9_21.4: <specific function> = specific_function %.loc9_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc9_21: init %i32 = call %.loc9_21.4(%.loc9_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc9_21.5: %i32 = value_of_initializer %int.convert_checked.loc9_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc9_21.6: %i32 = converted %.loc9_21.1, %.loc9_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc9_24.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_24.3: <bound method> = bound_method %.loc9_24.1, %.loc9_24.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_24.4: <specific function> = specific_function %.loc9_24.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc9_24: init %i32 = call %.loc9_24.4(%.loc9_24.1) [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_24.5: %i32 = value_of_initializer %int.convert_checked.loc9_24 [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_24.6: %i32 = converted %.loc9_24.1, %.loc9_24.5 [template = constants.%.36]
// CHECK:STDOUT:   %int.less.loc9: init bool = call %Less.ref.loc9(%.loc9_21.6, %.loc9_24.6) [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %int.less.loc9 [template = constants.%.37]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %int.less.loc9, %.loc9_13.1 [template = constants.%.37]
// CHECK:STDOUT:   if %.loc9_13.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
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
// CHECK:STDOUT:   %.loc9_13.3: type = block_arg !if.expr.result.loc9 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc10: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc10: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc10_22.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_25.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_22.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_22.3: <bound method> = bound_method %.loc10_22.1, %.loc10_22.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc10_22.4: <specific function> = specific_function %.loc10_22.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc10_22: init %i32 = call %.loc10_22.4(%.loc10_22.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_22.5: %i32 = value_of_initializer %int.convert_checked.loc10_22 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_22.6: %i32 = converted %.loc10_22.1, %.loc10_22.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_25.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_25.3: <bound method> = bound_method %.loc10_25.1, %.loc10_25.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc10_25.4: <specific function> = specific_function %.loc10_25.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc10_25: init %i32 = call %.loc10_25.4(%.loc10_25.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_25.5: %i32 = value_of_initializer %int.convert_checked.loc10_25 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_25.6: %i32 = converted %.loc10_25.1, %.loc10_25.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.less.loc10: init bool = call %Less.ref.loc10(%.loc10_22.6, %.loc10_25.6) [template = constants.%.38]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %int.less.loc10 [template = constants.%.38]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %int.less.loc10, %.loc10_14.1 [template = constants.%.38]
// CHECK:STDOUT:   if %.loc10_14.2 br !if.expr.then.loc10 else br !if.expr.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc10:
// CHECK:STDOUT:   %True.ref.loc10: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc10(%True.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc10:
// CHECK:STDOUT:   %False.ref.loc10: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc10(%False.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc10:
// CHECK:STDOUT:   %.loc10_14.3: type = block_arg !if.expr.result.loc10 [template = constants.%False]
// CHECK:STDOUT:   %false_.ref.loc11: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc11: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc11_22.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_25.1: Core.IntLiteral = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %.loc11_22.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_22.3: <bound method> = bound_method %.loc11_22.1, %.loc11_22.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_22.4: <specific function> = specific_function %.loc11_22.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc11_22: init %i32 = call %.loc11_22.4(%.loc11_22.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_22.5: %i32 = value_of_initializer %int.convert_checked.loc11_22 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_22.6: %i32 = converted %.loc11_22.1, %.loc11_22.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_25.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_25.3: <bound method> = bound_method %.loc11_25.1, %.loc11_25.2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_25.4: <specific function> = specific_function %.loc11_25.3, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc11_25: init %i32 = call %.loc11_25.4(%.loc11_25.1) [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_25.5: %i32 = value_of_initializer %int.convert_checked.loc11_25 [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_25.6: %i32 = converted %.loc11_25.1, %.loc11_25.5 [template = constants.%.42]
// CHECK:STDOUT:   %int.less.loc11: init bool = call %Less.ref.loc11(%.loc11_22.6, %.loc11_25.6) [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %int.less.loc11 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %int.less.loc11, %.loc11_14.1 [template = constants.%.38]
// CHECK:STDOUT:   if %.loc11_14.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %True.ref.loc11: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc11(%True.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %False.ref.loc11: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc11(%False.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_14.3: type = block_arg !if.expr.result.loc11 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc12: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc12: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_28.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_28.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_28.3: <bound method> = bound_method %.loc12_28.1, %.loc12_28.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_28.4: <specific function> = specific_function %.loc12_28.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc12_28: init %i32 = call %.loc12_28.4(%.loc12_28.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_28.5: %i32 = value_of_initializer %int.convert_checked.loc12_28 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_28.6: %i32 = converted %.loc12_28.1, %.loc12_28.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.snegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_28.6) [template = constants.%.43]
// CHECK:STDOUT:   %.loc12_32.1: Core.IntLiteral = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %.loc12_27.1: %i32 = value_of_initializer %int.snegate.loc12 [template = constants.%.43]
// CHECK:STDOUT:   %.loc12_27.2: %i32 = converted %int.snegate.loc12, %.loc12_27.1 [template = constants.%.43]
// CHECK:STDOUT:   %.loc12_32.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_32.3: <bound method> = bound_method %.loc12_32.1, %.loc12_32.2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc12_32.4: <specific function> = specific_function %.loc12_32.3, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc12_32: init %i32 = call %.loc12_32.4(%.loc12_32.1) [template = constants.%.42]
// CHECK:STDOUT:   %.loc12_32.5: %i32 = value_of_initializer %int.convert_checked.loc12_32 [template = constants.%.42]
// CHECK:STDOUT:   %.loc12_32.6: %i32 = converted %.loc12_32.1, %.loc12_32.5 [template = constants.%.42]
// CHECK:STDOUT:   %int.less.loc12: init bool = call %Less.ref.loc12(%.loc12_27.2, %.loc12_32.6) [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %int.less.loc12 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %int.less.loc12, %.loc12_13.1 [template = constants.%.37]
// CHECK:STDOUT:   if %.loc12_13.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %True.ref.loc12: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc12(%True.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %False.ref.loc12: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc12(%False.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_13.3: type = block_arg !if.expr.result.loc12 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc13: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc13: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc13_22.1: Core.IntLiteral = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %Negate.ref.loc13: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc13_32.1: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_32.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_32.3: <bound method> = bound_method %.loc13_32.1, %.loc13_32.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_32.4: <specific function> = specific_function %.loc13_32.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %i32 = call %.loc13_32.4(%.loc13_32.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_32.5: %i32 = value_of_initializer %int.convert_checked.loc13_32 [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_32.6: %i32 = converted %.loc13_32.1, %.loc13_32.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.snegate.loc13: init %i32 = call %Negate.ref.loc13(%.loc13_32.6) [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_22.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_22.3: <bound method> = bound_method %.loc13_22.1, %.loc13_22.2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc13_22.4: <specific function> = specific_function %.loc13_22.3, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc13_22: init %i32 = call %.loc13_22.4(%.loc13_22.1) [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_22.5: %i32 = value_of_initializer %int.convert_checked.loc13_22 [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_22.6: %i32 = converted %.loc13_22.1, %.loc13_22.5 [template = constants.%.42]
// CHECK:STDOUT:   %.loc13_31.1: %i32 = value_of_initializer %int.snegate.loc13 [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_31.2: %i32 = converted %int.snegate.loc13, %.loc13_31.1 [template = constants.%.43]
// CHECK:STDOUT:   %int.less.loc13: init bool = call %Less.ref.loc13(%.loc13_22.6, %.loc13_31.2) [template = constants.%.38]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %int.less.loc13 [template = constants.%.38]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %int.less.loc13, %.loc13_14.1 [template = constants.%.38]
// CHECK:STDOUT:   if %.loc13_14.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %True.ref.loc13: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc13(%True.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %False.ref.loc13: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc13(%False.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_14.3: type = block_arg !if.expr.result.loc13 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.less: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_20.1: bool = value_of_initializer %int.less
// CHECK:STDOUT:   %.loc17_20.2: bool = converted %int.less, %.loc17_20.1
// CHECK:STDOUT:   return %.loc17_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
