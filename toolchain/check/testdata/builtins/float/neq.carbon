// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/neq.carbon

// --- float_neq.carbon

fn Neq(a: f64, b: f64) -> bool = "float.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Neq(1.0, 2.0) then True else False);
  false_ as (if Neq(1.0, 1.0) then True else False);
}

fn RuntimeCallIsValid(a: f64, b: f64) -> bool {
  return Neq(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.neq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";

// CHECK:STDOUT: --- float_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [concrete]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [concrete]
// CHECK:STDOUT:   %True: type = class_type @True [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %False: type = class_type @False [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.f4e: f64 = float_literal 1 [concrete]
// CHECK:STDOUT:   %float.be3: f64 = float_literal 2 [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
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
// CHECK:STDOUT:     .Neq = %Neq.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCallIsValid = %RuntimeCallIsValid.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Neq.decl: %Neq.type = fn_decl @Neq [concrete = constants.%Neq] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %bool.make_type, %.loc2_27.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc2_11.1: type = splice_block %.loc2_11.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc2_11: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc2_11: init type = call constants.%Float(%int_64.loc2_11) [concrete = f64]
// CHECK:STDOUT:       %.loc2_11.2: type = value_of_initializer %float.make_type.loc2_11 [concrete = f64]
// CHECK:STDOUT:       %.loc2_11.3: type = converted %float.make_type.loc2_11, %.loc2_11.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc2_19.1: type = splice_block %.loc2_19.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc2_19: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc2_19: init type = call constants.%Float(%int_64.loc2_19) [concrete = f64]
// CHECK:STDOUT:       %.loc2_19.2: type = value_of_initializer %float.make_type.loc2_19 [concrete = f64]
// CHECK:STDOUT:       %.loc2_19.3: type = converted %float.make_type.loc2_19, %.loc2_19.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
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
// CHECK:STDOUT:     %True.ref.loc7: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param call_param1
// CHECK:STDOUT:     %False.ref.loc7: type = name_ref False, file.%False.decl [concrete = constants.%False]
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
// CHECK:STDOUT:     %.loc12_42.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc12_42.2: type = converted %bool.make_type, %.loc12_42.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc12_26.1: type = splice_block %.loc12_26.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc12_26: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc12_26: init type = call constants.%Float(%int_64.loc12_26) [concrete = f64]
// CHECK:STDOUT:       %.loc12_26.2: type = value_of_initializer %float.make_type.loc12_26 [concrete = f64]
// CHECK:STDOUT:       %.loc12_26.3: type = converted %float.make_type.loc12_26, %.loc12_26.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc12_34.1: type = splice_block %.loc12_34.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc12_34: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc12_34: init type = call constants.%Float(%int_64.loc12_34) [concrete = f64]
// CHECK:STDOUT:       %.loc12_34.2: type = value_of_initializer %float.make_type.loc12_34 [concrete = f64]
// CHECK:STDOUT:       %.loc12_34.3: type = converted %float.make_type.loc12_34, %.loc12_34.2 [concrete = f64]
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
// CHECK:STDOUT: fn @Neq(%a.param_patt: f64, %b.param_patt: f64) -> bool = "float.neq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Neq.ref.loc8: %Neq.type = name_ref Neq, file.%Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:   %float.loc8_20: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.loc8_25: f64 = float_literal 2 [concrete = constants.%float.be3]
// CHECK:STDOUT:   %float.neq.loc8: init bool = call %Neq.ref.loc8(%float.loc8_20, %float.loc8_25) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_13.1: bool = value_of_initializer %float.neq.loc8 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_13.2: bool = converted %float.neq.loc8, %.loc8_13.1 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc8_13.2 br !if.expr.then.loc8 else br !if.expr.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8:
// CHECK:STDOUT:   %True.ref.loc8: type = name_ref True, file.%True.decl [concrete = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc8(%True.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8:
// CHECK:STDOUT:   %False.ref.loc8: type = name_ref False, file.%False.decl [concrete = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc8(%False.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8:
// CHECK:STDOUT:   %.loc8_13.3: type = block_arg !if.expr.result.loc8 [concrete = constants.%True]
// CHECK:STDOUT:   %false_.ref: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Neq.ref.loc9: %Neq.type = name_ref Neq, file.%Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:   %float.loc9_21: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.loc9_26: f64 = float_literal 1 [concrete = constants.%float.f4e]
// CHECK:STDOUT:   %float.neq.loc9: init bool = call %Neq.ref.loc9(%float.loc9_21, %float.loc9_26) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_14.1: bool = value_of_initializer %float.neq.loc9 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_14.2: bool = converted %float.neq.loc9, %.loc9_14.1 [concrete = constants.%false]
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
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param_patt: f64, %b.param_patt: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: %Neq.type = name_ref Neq, file.%Neq.decl [concrete = constants.%Neq]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.neq: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc13_19.1: bool = value_of_initializer %float.neq
// CHECK:STDOUT:   %.loc13_19.2: bool = converted %float.neq, %.loc13_19.1
// CHECK:STDOUT:   return %.loc13_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %WrongResult.type: type = fn_type @WrongResult [concrete]
// CHECK:STDOUT:   %WrongResult: %WrongResult.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrongResult = %WrongResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongResult.decl: %WrongResult.type = fn_decl @WrongResult [concrete = constants.%WrongResult] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc8_35: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc8_35: init type = call constants.%Float(%int_64.loc8_35) [concrete = f64]
// CHECK:STDOUT:     %.loc8_35.1: type = value_of_initializer %float.make_type.loc8_35 [concrete = f64]
// CHECK:STDOUT:     %.loc8_35.2: type = converted %float.make_type.loc8_35, %.loc8_35.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc8_19.1: type = splice_block %.loc8_19.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc8_19: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc8_19: init type = call constants.%Float(%int_64.loc8_19) [concrete = f64]
// CHECK:STDOUT:       %.loc8_19.2: type = value_of_initializer %float.make_type.loc8_19 [concrete = f64]
// CHECK:STDOUT:       %.loc8_19.3: type = converted %float.make_type.loc8_19, %.loc8_19.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc8_27.1: type = splice_block %.loc8_27.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc8_27: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc8_27: init type = call constants.%Float(%int_64.loc8_27) [concrete = f64]
// CHECK:STDOUT:       %.loc8_27.2: type = value_of_initializer %float.make_type.loc8_27 [concrete = f64]
// CHECK:STDOUT:       %.loc8_27.3: type = converted %float.make_type.loc8_27, %.loc8_27.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongResult(%a.param_patt: f64, %b.param_patt: f64) -> f64;
// CHECK:STDOUT:
