// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/less_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/less_eq.carbon

// --- int_less_eq.carbon

fn LessEq(a: i32, b: i32) -> bool = "int.less_eq";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if LessEq(1, 2) then True else False);
  true_ as (if LessEq(1, 1) then True else False);
  false_ as (if LessEq(1, 0) then True else False);
  true_ as (if LessEq(Negate(1), 0) then True else False);
  false_ as (if LessEq(0, Negate(1)) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return LessEq(a, b);
}

// CHECK:STDOUT: --- int_less_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [template]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .LessEq = %LessEq.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LessEq.decl: %LessEq.type = fn_decl @LessEq [template = constants.%LessEq] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_14: init type = call constants.%Int(%int_32.loc2_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.1: type = value_of_initializer %int.make_type_signed.loc2_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.2: type = converted %int.make_type_signed.loc2_14, %.loc2_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_22: init type = call constants.%Int(%int_32.loc2_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.1: type = value_of_initializer %int.make_type_signed.loc2_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.2: type = converted %int.make_type_signed.loc2_22, %.loc2_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc2_30.2: type = converted %bool.make_type, %.loc2_30.1 [template = bool]
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
// CHECK:STDOUT:     %int_32.loc3_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_14: init type = call constants.%Int(%int_32.loc3_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_14.1: type = value_of_initializer %int.make_type_signed.loc3_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_14.2: type = converted %int.make_type_signed.loc3_14, %.loc3_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc3_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_22: init type = call constants.%Int(%int_32.loc3_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_22.1: type = value_of_initializer %int.make_type_signed.loc3_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_22.2: type = converted %int.make_type_signed.loc3_22, %.loc3_22.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc16_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16_19: init type = call constants.%Int(%int_32.loc16_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.1: type = value_of_initializer %int.make_type_signed.loc16_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_19.2: type = converted %int.make_type_signed.loc16_19, %.loc16_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc16_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc16_27: init type = call constants.%Int(%int_32.loc16_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.1: type = value_of_initializer %int.make_type_signed.loc16_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %int.make_type_signed.loc16_27, %.loc16_27.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessEq(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.less_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc9: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc9_23: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9_23: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_23: <specific function> = specific_function %Convert.bound.loc9_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_23: init %i32 = call %Convert.specific_fn.loc9_23(%int_1.loc9) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_23.1: %i32 = value_of_initializer %int.convert_checked.loc9_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_23.2: %i32 = converted %int_1.loc9, %.loc9_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc9_26: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9_26: <bound method> = bound_method %int_2, %impl.elem0.loc9_26 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_26: <specific function> = specific_function %Convert.bound.loc9_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_26: init %i32 = call %Convert.specific_fn.loc9_26(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc9_26.1: %i32 = value_of_initializer %int.convert_checked.loc9_26 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc9_26.2: %i32 = converted %int_2, %.loc9_26.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.less_eq.loc9: init bool = call %LessEq.ref.loc9(%.loc9_23.2, %.loc9_26.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %int.less_eq.loc9 [template = constants.%true]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %int.less_eq.loc9, %.loc9_13.1 [template = constants.%true]
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
// CHECK:STDOUT:   %true_.ref.loc10: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc10: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc10_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc10_26: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc10_23: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_23: <bound method> = bound_method %int_1.loc10_23, %impl.elem0.loc10_23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_23: <specific function> = specific_function %Convert.bound.loc10_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_23: init %i32 = call %Convert.specific_fn.loc10_23(%int_1.loc10_23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %int.convert_checked.loc10_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int_1.loc10_23, %.loc10_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc10_26: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_26: <bound method> = bound_method %int_1.loc10_26, %impl.elem0.loc10_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_26: <specific function> = specific_function %Convert.bound.loc10_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_26: init %i32 = call %Convert.specific_fn.loc10_26(%int_1.loc10_26) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_26.1: %i32 = value_of_initializer %int.convert_checked.loc10_26 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_26.2: %i32 = converted %int_1.loc10_26, %.loc10_26.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.less_eq.loc10: init bool = call %LessEq.ref.loc10(%.loc10_23.2, %.loc10_26.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_13.1: bool = value_of_initializer %int.less_eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_13.2: bool = converted %int.less_eq.loc10, %.loc10_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc10_13.2 br !if.expr.then.loc10 else br !if.expr.else.loc10
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
// CHECK:STDOUT:   %.loc10_13.3: type = block_arg !if.expr.result.loc10 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc11: %False = name_ref false_, %false_
// CHECK:STDOUT:   %LessEq.ref.loc11: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_24: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_24: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_24 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_24: <specific function> = specific_function %Convert.bound.loc11_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_24: init %i32 = call %Convert.specific_fn.loc11_24(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_24.1: %i32 = value_of_initializer %int.convert_checked.loc11_24 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_24.2: %i32 = converted %int_1.loc11, %.loc11_24.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_27: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11_27 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27: <specific function> = specific_function %Convert.bound.loc11_27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_27: init %i32 = call %Convert.specific_fn.loc11_27(%int_0.loc11) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_27.1: %i32 = value_of_initializer %int.convert_checked.loc11_27 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_27.2: %i32 = converted %int_0.loc11, %.loc11_27.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.less_eq.loc11: init bool = call %LessEq.ref.loc11(%.loc11_24.2, %.loc11_27.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %int.less_eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %int.less_eq.loc11, %.loc11_14.1 [template = constants.%false]
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
// CHECK:STDOUT:   %LessEq.ref.loc12: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_30: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_30: <specific function> = specific_function %Convert.bound.loc12_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_30: init %i32 = call %Convert.specific_fn.loc12_30(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_30.1: %i32 = value_of_initializer %int.convert_checked.loc12_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_30.2: %i32 = converted %int_1.loc12, %.loc12_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_30.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc12_31.1: %i32 = value_of_initializer %int.snegate.loc12 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_31.2: %i32 = converted %int.snegate.loc12, %.loc12_31.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_34: <bound method> = bound_method %int_0.loc12, %impl.elem0.loc12_34 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_34: <specific function> = specific_function %Convert.bound.loc12_34, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_34: init %i32 = call %Convert.specific_fn.loc12_34(%int_0.loc12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %int.convert_checked.loc12_34 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_0.loc12, %.loc12_34.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.less_eq.loc12: init bool = call %LessEq.ref.loc12(%.loc12_31.2, %.loc12_34.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %int.less_eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %int.less_eq.loc12, %.loc12_13.1 [template = constants.%true]
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
// CHECK:STDOUT:   %LessEq.ref.loc13: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %Negate.ref.loc13: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_34: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_34 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_34: <specific function> = specific_function %Convert.bound.loc13_34, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_34: init %i32 = call %Convert.specific_fn.loc13_34(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %int.convert_checked.loc13_34 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_1.loc13, %.loc13_34.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc13: init %i32 = call %Negate.ref.loc13(%.loc13_34.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc13_24: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_24: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13_24 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_24: <specific function> = specific_function %Convert.bound.loc13_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_24: init %i32 = call %Convert.specific_fn.loc13_24(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.1: %i32 = value_of_initializer %int.convert_checked.loc13_24 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_24.2: %i32 = converted %int_0.loc13, %.loc13_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %int.snegate.loc13 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %int.snegate.loc13, %.loc13_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.less_eq.loc13: init bool = call %LessEq.ref.loc13(%.loc13_24.2, %.loc13_35.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %int.less_eq.loc13 [template = constants.%false]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %int.less_eq.loc13, %.loc13_14.1 [template = constants.%false]
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
// CHECK:STDOUT:   %LessEq.ref: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.less_eq: init bool = call %LessEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_22.1: bool = value_of_initializer %int.less_eq
// CHECK:STDOUT:   %.loc17_22.2: bool = converted %int.less_eq, %.loc17_22.1
// CHECK:STDOUT:   return %.loc17_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
