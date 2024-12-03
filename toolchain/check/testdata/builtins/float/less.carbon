// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/less.carbon

// --- float_less.carbon

fn Less(a: f64, b: f64) -> bool = "float.less";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1.0, 2.0) then True else False);
  false_ as (if Less(1.0, 1.0) then True else False);
  false_ as (if Less(1.0, 0.0) then True else False);
  true_ as (if Less(Negate(1.0), 0.0) then True else False);
  false_ as (if Less(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCall(a: f64, b: f64) -> bool {
  return Less(a, b);
}

// CHECK:STDOUT: --- float_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [template]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %float.1: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %float.2: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %float.3: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %float.4: f64 = float_literal -1 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
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
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc2_12: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_12: init type = call constants.%Float(%int_64.loc2_12) [template = f64]
// CHECK:STDOUT:     %.loc2_12.1: type = value_of_initializer %float.make_type.loc2_12 [template = f64]
// CHECK:STDOUT:     %.loc2_12.2: type = converted %float.make_type.loc2_12, %.loc2_12.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc2_20: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_20: init type = call constants.%Float(%int_64.loc2_20) [template = f64]
// CHECK:STDOUT:     %.loc2_20.1: type = value_of_initializer %float.make_type.loc2_20 [template = f64]
// CHECK:STDOUT:     %.loc2_20.2: type = converted %float.make_type.loc2_20, %.loc2_20.1 [template = f64]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_28.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc2_28.2: type = converted %bool.make_type, %.loc2_28.1 [template = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc3_14: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc3_14: init type = call constants.%Float(%int_64.loc3_14) [template = f64]
// CHECK:STDOUT:     %.loc3_14.1: type = value_of_initializer %float.make_type.loc3_14 [template = f64]
// CHECK:STDOUT:     %.loc3_14.2: type = converted %float.make_type.loc3_14, %.loc3_14.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc3_22: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc3_22: init type = call constants.%Float(%int_64.loc3_22) [template = f64]
// CHECK:STDOUT:     %.loc3_22.1: type = value_of_initializer %float.make_type.loc3_22 [template = f64]
// CHECK:STDOUT:     %.loc3_22.2: type = converted %float.make_type.loc3_22, %.loc3_22.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
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
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc16_19: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc16_19: init type = call constants.%Float(%int_64.loc16_19) [template = f64]
// CHECK:STDOUT:     %.loc16_19.1: type = value_of_initializer %float.make_type.loc16_19 [template = f64]
// CHECK:STDOUT:     %.loc16_19.2: type = converted %float.make_type.loc16_19, %.loc16_19.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc16_27: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc16_27: init type = call constants.%Float(%int_64.loc16_27) [template = f64]
// CHECK:STDOUT:     %.loc16_27.1: type = value_of_initializer %float.make_type.loc16_27 [template = f64]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %float.make_type.loc16_27, %.loc16_27.1 [template = f64]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc16_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc16_35.2: type = converted %bool.make_type, %.loc16_35.1 [template = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
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
// CHECK:STDOUT: fn @Less(%a.param_patt: f64, %b.param_patt: f64) -> bool = "float.less";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc9: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %float.loc9_21: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.loc9_26: f64 = float_literal 2 [template = constants.%float.2]
// CHECK:STDOUT:   %float.less.loc9: init bool = call %Less.ref.loc9(%float.loc9_21, %float.loc9_26) [template = constants.%true]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %float.less.loc9 [template = constants.%true]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %float.less.loc9, %.loc9_13.1 [template = constants.%true]
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
// CHECK:STDOUT:   %float.loc10_22: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.loc10_27: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.less.loc10: init bool = call %Less.ref.loc10(%float.loc10_22, %float.loc10_27) [template = constants.%false]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %float.less.loc10 [template = constants.%false]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %float.less.loc10, %.loc10_14.1 [template = constants.%false]
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
// CHECK:STDOUT:   %float.loc11_22: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.loc11_27: f64 = float_literal 0 [template = constants.%float.3]
// CHECK:STDOUT:   %float.less.loc11: init bool = call %Less.ref.loc11(%float.loc11_22, %float.loc11_27) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %float.less.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %float.less.loc11, %.loc11_14.1 [template = constants.%false]
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
// CHECK:STDOUT:   %float.loc12_28: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.negate.loc12: init f64 = call %Negate.ref.loc12(%float.loc12_28) [template = constants.%float.4]
// CHECK:STDOUT:   %float.loc12_34: f64 = float_literal 0 [template = constants.%float.3]
// CHECK:STDOUT:   %.loc12_31.1: f64 = value_of_initializer %float.negate.loc12 [template = constants.%float.4]
// CHECK:STDOUT:   %.loc12_31.2: f64 = converted %float.negate.loc12, %.loc12_31.1 [template = constants.%float.4]
// CHECK:STDOUT:   %float.less.loc12: init bool = call %Less.ref.loc12(%.loc12_31.2, %float.loc12_34) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %float.less.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %float.less.loc12, %.loc12_13.1 [template = constants.%true]
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
// CHECK:STDOUT:   %float.loc13_22: f64 = float_literal 0 [template = constants.%float.3]
// CHECK:STDOUT:   %Negate.ref.loc13: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %float.loc13_34: f64 = float_literal 1 [template = constants.%float.1]
// CHECK:STDOUT:   %float.negate.loc13: init f64 = call %Negate.ref.loc13(%float.loc13_34) [template = constants.%float.4]
// CHECK:STDOUT:   %.loc13_37.1: f64 = value_of_initializer %float.negate.loc13 [template = constants.%float.4]
// CHECK:STDOUT:   %.loc13_37.2: f64 = converted %float.negate.loc13, %.loc13_37.1 [template = constants.%float.4]
// CHECK:STDOUT:   %float.less.loc13: init bool = call %Less.ref.loc13(%float.loc13_22, %.loc13_37.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %float.less.loc13 [template = constants.%false]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %float.less.loc13, %.loc13_14.1 [template = constants.%false]
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
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: f64, %b.param_patt: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.less: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_20.1: bool = value_of_initializer %float.less
// CHECK:STDOUT:   %.loc17_20.2: bool = converted %float.less, %.loc17_20.1
// CHECK:STDOUT:   return %.loc17_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
