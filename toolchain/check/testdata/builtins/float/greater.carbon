// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/greater.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/greater.carbon

// --- float_greater.carbon

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Greater(1.0, 2.0) then True else False);
  false_ as (if Greater(1.0, 1.0) then True else False);
  true_ as (if Greater(1.0, 0.0) then True else False);
  false_ as (if Greater(Negate(1.0), 0.0) then True else False);
  true_ as (if Greater(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  return Greater(a, b);
}

// CHECK:STDOUT: --- float_greater.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Greater.type: type = fn_type @Greater [concrete]
// CHECK:STDOUT:   %Greater: %Greater.type = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %True: type = class_type @True [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %False: type = class_type @False [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.f4e: f64 = float_literal 1 [concrete]
// CHECK:STDOUT:   %float.be3: f64 = float_literal 2 [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %float.555: f64 = float_literal 0 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %float.f51: f64 = float_literal -1 [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValid.type: type = fn_type @RuntimeCallIsValid [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValid: %RuntimeCallIsValid.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Greater = %Greater.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCallIsValid = %RuntimeCallIsValid.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Greater.decl: %Greater.type = fn_decl @Greater [concrete = constants.%Greater] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc2_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc2_31.2: type = converted %bool.make_type, %.loc2_31.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc2_15.1: type = splice_block %.loc2_15.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc2_15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc2_15: init type = call constants.%Float(%int_64.loc2_15) [concrete = f64]
// CHECK:STDOUT:       %.loc2_15.2: type = value_of_initializer %float.make_type.loc2_15 [concrete = f64]
// CHECK:STDOUT:       %.loc2_15.3: type = converted %float.make_type.loc2_15, %.loc2_15.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc2_23.1: type = splice_block %.loc2_23.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc2_23: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc2_23: init type = call constants.%Float(%int_64.loc2_23) [concrete = f64]
// CHECK:STDOUT:       %.loc2_23.2: type = value_of_initializer %float.make_type.loc2_23 [concrete = f64]
// CHECK:STDOUT:       %.loc2_23.3: type = converted %float.make_type.loc2_23, %.loc2_23.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [concrete = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc3_22: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc3_22: init type = call constants.%Float(%int_64.loc3_22) [concrete = f64]
// CHECK:STDOUT:     %.loc3_22.1: type = value_of_initializer %float.make_type.loc3_22 [concrete = f64]
// CHECK:STDOUT:     %.loc3_22.2: type = converted %float.make_type.loc3_22, %.loc3_22.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc3_14.1: type = splice_block %.loc3_14.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc3_14: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc3_14: init type = call constants.%Float(%int_64.loc3_14) [concrete = f64]
// CHECK:STDOUT:       %.loc3_14.2: type = value_of_initializer %float.make_type.loc3_14 [concrete = f64]
// CHECK:STDOUT:       %.loc3_14.3: type = converted %float.make_type.loc3_14, %.loc3_14.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [concrete = constants.%True] {} {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [concrete = constants.%False] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %true_.patt: %True = binding_pattern true_
// CHECK:STDOUT:     %true_.param_patt: %True = value_param_pattern %true_.patt, call_param0
// CHECK:STDOUT:     %false_.patt: %False = binding_pattern false_
// CHECK:STDOUT:     %false_.param_patt: %False = value_param_pattern %false_.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %true_.param: %True = value_param call_param0
// CHECK:STDOUT:     %True.ref.loc8: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param call_param1
// CHECK:STDOUT:     %False.ref.loc8: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:     %false_: %False = bind_name false_, %false_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallIsValid.decl: %RuntimeCallIsValid.type = fn_decl @RuntimeCallIsValid [concrete = constants.%RuntimeCallIsValid] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_42.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc16_42.2: type = converted %bool.make_type, %.loc16_42.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc16_26.1: type = splice_block %.loc16_26.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc16_26: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc16_26: init type = call constants.%Float(%int_64.loc16_26) [concrete = f64]
// CHECK:STDOUT:       %.loc16_26.2: type = value_of_initializer %float.make_type.loc16_26 [concrete = f64]
// CHECK:STDOUT:       %.loc16_26.3: type = converted %float.make_type.loc16_26, %.loc16_26.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc16_34.1: type = splice_block %.loc16_34.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc16_34: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc16_34: init type = call constants.%Float(%int_64.loc16_34) [concrete = f64]
// CHECK:STDOUT:       %.loc16_34.2: type = value_of_initializer %float.make_type.loc16_34 [concrete = f64]
// CHECK:STDOUT:       %.loc16_34.3: type = converted %float.make_type.loc16_34, %.loc16_34.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Greater(%a.param_patt: f64, %b.param_patt: f64) -> bool = "float.greater";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false_.ref.loc9: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc9: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %float.loc9_25: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.loc9_30: f64 = float_literal 2 [concrete = constants.%float.be3]
// CHECK:STDOUT:   %float.greater.loc9: init bool = call %Greater.ref.loc9(%float.loc9_25, %float.loc9_30) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_14.1: bool = value_of_initializer %float.greater.loc9 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_14.2: bool = converted %float.greater.loc9, %.loc9_14.1 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc9_14.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %True.ref.loc9: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc9(%True.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %False.ref.loc9: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc9(%False.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_14.3: type = block_arg !if.expr.result.loc9 [concrete = constants.%False]
// CHECK:STDOUT:   %false_.ref.loc10: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc10: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %float.loc10_25: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.loc10_30: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.greater.loc10: init bool = call %Greater.ref.loc10(%float.loc10_25, %float.loc10_30) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %float.greater.loc10 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %float.greater.loc10, %.loc10_14.1 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc10_14.2 br !if.expr.then.loc10 else br !if.expr.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc10:
// CHECK:STDOUT:   %True.ref.loc10: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc10(%True.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc10:
// CHECK:STDOUT:   %False.ref.loc10: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc10(%False.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc10:
// CHECK:STDOUT:   %.loc10_14.3: type = block_arg !if.expr.result.loc10 [concrete = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc11: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Greater.ref.loc11: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %float.loc11_24: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.loc11_29: f64 = float_literal 0 [concrete = constants.%float.555]
// CHECK:STDOUT:   %float.greater.loc11: init bool = call %Greater.ref.loc11(%float.loc11_24, %float.loc11_29) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc11_13.1: bool = value_of_initializer %float.greater.loc11 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc11_13.2: bool = converted %float.greater.loc11, %.loc11_13.1 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc11_13.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %True.ref.loc11: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc11(%True.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %False.ref.loc11: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc11(%False.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_13.3: type = block_arg !if.expr.result.loc11 [concrete = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc12: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc12: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc12_32: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.negate.loc12: init f64 = call %Negate.ref.loc12(%float.loc12_32) [concrete = constants.%float.f51]
// CHECK:STDOUT:   %float.loc12_38: f64 = float_literal 0 [concrete = constants.%float.555]
// CHECK:STDOUT:   %.loc12_35.1: f64 = value_of_initializer %float.negate.loc12 [concrete = constants.%float.f51]
// CHECK:STDOUT:   %.loc12_35.2: f64 = converted %float.negate.loc12, %.loc12_35.1 [concrete = constants.%float.f51]
// CHECK:STDOUT:   %float.greater.loc12: init bool = call %Greater.ref.loc12(%.loc12_35.2, %float.loc12_38) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc12_14.1: bool = value_of_initializer %float.greater.loc12 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc12_14.2: bool = converted %float.greater.loc12, %.loc12_14.1 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc12_14.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %True.ref.loc12: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc12(%True.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %False.ref.loc12: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc12(%False.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_14.3: type = block_arg !if.expr.result.loc12 [concrete = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc13: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Greater.ref.loc13: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %float.loc13_24: f64 = float_literal 0 [concrete = constants.%float.555]
// CHECK:STDOUT:   %Negate.ref.loc13: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float.loc13_36: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.negate.loc13: init f64 = call %Negate.ref.loc13(%float.loc13_36) [concrete = constants.%float.f51]
// CHECK:STDOUT:   %.loc13_39.1: f64 = value_of_initializer %float.negate.loc13 [concrete = constants.%float.f51]
// CHECK:STDOUT:   %.loc13_39.2: f64 = converted %float.negate.loc13, %.loc13_39.1 [concrete = constants.%float.f51]
// CHECK:STDOUT:   %float.greater.loc13: init bool = call %Greater.ref.loc13(%float.loc13_24, %.loc13_39.2) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc13_13.1: bool = value_of_initializer %float.greater.loc13 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc13_13.2: bool = converted %float.greater.loc13, %.loc13_13.1 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc13_13.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %True.ref.loc13: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc13(%True.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %False.ref.loc13: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc13(%False.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_13.3: type = block_arg !if.expr.result.loc13 [concrete = constants.%True]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param_patt: f64, %b.param_patt: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Greater.ref: %Greater.type = name_ref Greater, file.%Greater.decl [concrete = constants.%Greater]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.greater: init bool = call %Greater.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_23.1: bool = value_of_initializer %float.greater
// CHECK:STDOUT:   %.loc17_23.2: bool = converted %float.greater, %.loc17_23.1
// CHECK:STDOUT:   return %.loc17_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
