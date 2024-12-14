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
  // CHECK:STDERR: int_less_eq.carbon:[[@LINE+4]]:12: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   true_ as (if LessEq(1, 2) then True else False);
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  true_ as (if LessEq(1, 2) then True else False);
  // CHECK:STDERR: int_less_eq.carbon:[[@LINE+4]]:12: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   true_ as (if LessEq(1, 1) then True else False);
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  true_ as (if LessEq(1, 1) then True else False);
  // CHECK:STDERR: int_less_eq.carbon:[[@LINE+4]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   false_ as (if LessEq(1, 0) then True else False);
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  false_ as (if LessEq(1, 0) then True else False);
  // CHECK:STDERR: int_less_eq.carbon:[[@LINE+4]]:12: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   true_ as (if LessEq(Negate(1), 0) then True else False);
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  true_ as (if LessEq(Negate(1), 0) then True else False);
  // CHECK:STDERR: int_less_eq.carbon:[[@LINE+3]]:13: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   false_ as (if LessEq(0, Negate(1)) then True else False);
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3), @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.3(%int_0) [symbolic]
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
// CHECK:STDOUT:     %int_32.loc35_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc35_19: init type = call constants.%Int(%int_32.loc35_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc35_19.1: type = value_of_initializer %int.make_type_signed.loc35_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc35_19.2: type = converted %int.make_type_signed.loc35_19, %.loc35_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc35_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc35_27: init type = call constants.%Int(%int_32.loc35_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc35_27.1: type = value_of_initializer %int.make_type_signed.loc35_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc35_27.2: type = converted %int.make_type_signed.loc35_27, %.loc35_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc35_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc35_35.2: type = converted %bool.make_type, %.loc35_35.1 [template = bool]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
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
// CHECK:STDOUT:   %true_.ref.loc13: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc13: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc13_23: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_23: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_23 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_23: <specific function> = specific_function %Convert.bound.loc13_23, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_23: init %iN = call %Convert.specific_fn.loc13_23(%int_1.loc13) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_23.1: %i32 = value_of_initializer %int.convert_checked.loc13_23 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_23.2: %i32 = converted %int_1.loc13, %.loc13_23.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc13_26: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_26: <bound method> = bound_method %int_2, %impl.elem0.loc13_26 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_26: <specific function> = specific_function %Convert.bound.loc13_26, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_26: init %iN = call %Convert.specific_fn.loc13_26(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_26.1: %i32 = value_of_initializer %int.convert_checked.loc13_26 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_26.2: %i32 = converted %int_2, %.loc13_26.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.less_eq.loc13: init bool = call %LessEq.ref.loc13(%.loc13_23.2, %.loc13_26.2)
// CHECK:STDOUT:   %.loc13_13.1: bool = value_of_initializer %int.less_eq.loc13
// CHECK:STDOUT:   %.loc13_13.2: bool = converted %int.less_eq.loc13, %.loc13_13.1
// CHECK:STDOUT:   if %.loc13_13.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
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
// CHECK:STDOUT:   %.loc13_13.3: type = block_arg !if.expr.result.loc13
// CHECK:STDOUT:   %true_.ref.loc18: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc18: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc18_23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc18_26: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc18_23: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_23: <bound method> = bound_method %int_1.loc18_23, %impl.elem0.loc18_23 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_23: <specific function> = specific_function %Convert.bound.loc18_23, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_23: init %iN = call %Convert.specific_fn.loc18_23(%int_1.loc18_23) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_23.1: %i32 = value_of_initializer %int.convert_checked.loc18_23 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_23.2: %i32 = converted %int_1.loc18_23, %.loc18_23.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc18_26: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_26: <bound method> = bound_method %int_1.loc18_26, %impl.elem0.loc18_26 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_26: <specific function> = specific_function %Convert.bound.loc18_26, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_26: init %iN = call %Convert.specific_fn.loc18_26(%int_1.loc18_26) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_26.1: %i32 = value_of_initializer %int.convert_checked.loc18_26 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc18_26.2: %i32 = converted %int_1.loc18_26, %.loc18_26.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.less_eq.loc18: init bool = call %LessEq.ref.loc18(%.loc18_23.2, %.loc18_26.2)
// CHECK:STDOUT:   %.loc18_13.1: bool = value_of_initializer %int.less_eq.loc18
// CHECK:STDOUT:   %.loc18_13.2: bool = converted %int.less_eq.loc18, %.loc18_13.1
// CHECK:STDOUT:   if %.loc18_13.2 br !if.expr.then.loc18 else br !if.expr.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc18:
// CHECK:STDOUT:   %True.ref.loc18: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc18(%True.ref.loc18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc18:
// CHECK:STDOUT:   %False.ref.loc18: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc18(%False.ref.loc18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc18:
// CHECK:STDOUT:   %.loc18_13.3: type = block_arg !if.expr.result.loc18
// CHECK:STDOUT:   %false_.ref.loc23: %False = name_ref false_, %false_
// CHECK:STDOUT:   %LessEq.ref.loc23: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc23_24: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc23_24: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23_24 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc23_24: <specific function> = specific_function %Convert.bound.loc23_24, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc23_24: init %iN = call %Convert.specific_fn.loc23_24(%int_1.loc23) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc23_24.1: %i32 = value_of_initializer %int.convert_checked.loc23_24 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc23_24.2: %i32 = converted %int_1.loc23, %.loc23_24.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc23_27: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc23_27: <bound method> = bound_method %int_0.loc23, %impl.elem0.loc23_27 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc23_27: <specific function> = specific_function %Convert.bound.loc23_27, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc23_27: init %iN = call %Convert.specific_fn.loc23_27(%int_0.loc23) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc23_27.1: %i32 = value_of_initializer %int.convert_checked.loc23_27 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc23_27.2: %i32 = converted %int_0.loc23, %.loc23_27.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.less_eq.loc23: init bool = call %LessEq.ref.loc23(%.loc23_24.2, %.loc23_27.2)
// CHECK:STDOUT:   %.loc23_14.1: bool = value_of_initializer %int.less_eq.loc23
// CHECK:STDOUT:   %.loc23_14.2: bool = converted %int.less_eq.loc23, %.loc23_14.1
// CHECK:STDOUT:   if %.loc23_14.2 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %True.ref.loc23: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc23(%True.ref.loc23)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %False.ref.loc23: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc23(%False.ref.loc23)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_14.3: type = block_arg !if.expr.result.loc23
// CHECK:STDOUT:   %true_.ref.loc28: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc28: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %Negate.ref.loc28: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc28_30: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc28_30: <bound method> = bound_method %int_1.loc28, %impl.elem0.loc28_30 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc28_30: <specific function> = specific_function %Convert.bound.loc28_30, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc28_30: init %iN = call %Convert.specific_fn.loc28_30(%int_1.loc28) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc28_30.1: %i32 = value_of_initializer %int.convert_checked.loc28_30 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc28_30.2: %i32 = converted %int_1.loc28, %.loc28_30.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc28: init %i32 = call %Negate.ref.loc28(%.loc28_30.2)
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc28_31.1: %i32 = value_of_initializer %int.snegate.loc28
// CHECK:STDOUT:   %.loc28_31.2: %i32 = converted %int.snegate.loc28, %.loc28_31.1
// CHECK:STDOUT:   %impl.elem0.loc28_34: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc28_34: <bound method> = bound_method %int_0.loc28, %impl.elem0.loc28_34 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc28_34: <specific function> = specific_function %Convert.bound.loc28_34, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc28_34: init %iN = call %Convert.specific_fn.loc28_34(%int_0.loc28) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc28_34.1: %i32 = value_of_initializer %int.convert_checked.loc28_34 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc28_34.2: %i32 = converted %int_0.loc28, %.loc28_34.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.less_eq.loc28: init bool = call %LessEq.ref.loc28(%.loc28_31.2, %.loc28_34.2)
// CHECK:STDOUT:   %.loc28_13.1: bool = value_of_initializer %int.less_eq.loc28
// CHECK:STDOUT:   %.loc28_13.2: bool = converted %int.less_eq.loc28, %.loc28_13.1
// CHECK:STDOUT:   if %.loc28_13.2 br !if.expr.then.loc28 else br !if.expr.else.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc28:
// CHECK:STDOUT:   %True.ref.loc28: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc28(%True.ref.loc28)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc28:
// CHECK:STDOUT:   %False.ref.loc28: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc28(%False.ref.loc28)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc28:
// CHECK:STDOUT:   %.loc28_13.3: type = block_arg !if.expr.result.loc28
// CHECK:STDOUT:   %false_.ref.loc32: %False = name_ref false_, %false_
// CHECK:STDOUT:   %LessEq.ref.loc32: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Negate.ref.loc32: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc32_34: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc32_34: <bound method> = bound_method %int_1.loc32, %impl.elem0.loc32_34 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc32_34: <specific function> = specific_function %Convert.bound.loc32_34, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc32_34: init %iN = call %Convert.specific_fn.loc32_34(%int_1.loc32) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc32_34.1: %i32 = value_of_initializer %int.convert_checked.loc32_34 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc32_34.2: %i32 = converted %int_1.loc32, %.loc32_34.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc32: init %i32 = call %Negate.ref.loc32(%.loc32_34.2)
// CHECK:STDOUT:   %impl.elem0.loc32_24: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc32_24: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32_24 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc32_24: <specific function> = specific_function %Convert.bound.loc32_24, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc32_24: init %iN = call %Convert.specific_fn.loc32_24(%int_0.loc32) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc32_24.1: %i32 = value_of_initializer %int.convert_checked.loc32_24 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc32_24.2: %i32 = converted %int_0.loc32, %.loc32_24.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc32_35.1: %i32 = value_of_initializer %int.snegate.loc32
// CHECK:STDOUT:   %.loc32_35.2: %i32 = converted %int.snegate.loc32, %.loc32_35.1
// CHECK:STDOUT:   %int.less_eq.loc32: init bool = call %LessEq.ref.loc32(%.loc32_24.2, %.loc32_35.2)
// CHECK:STDOUT:   %.loc32_14.1: bool = value_of_initializer %int.less_eq.loc32
// CHECK:STDOUT:   %.loc32_14.2: bool = converted %int.less_eq.loc32, %.loc32_14.1
// CHECK:STDOUT:   if %.loc32_14.2 br !if.expr.then.loc32 else br !if.expr.else.loc32
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc32:
// CHECK:STDOUT:   %True.ref.loc32: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc32(%True.ref.loc32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc32:
// CHECK:STDOUT:   %False.ref.loc32: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc32(%False.ref.loc32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc32:
// CHECK:STDOUT:   %.loc32_14.3: type = block_arg !if.expr.result.loc32
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LessEq.ref: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.less_eq: init bool = call %LessEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc36_22.1: bool = value_of_initializer %int.less_eq
// CHECK:STDOUT:   %.loc36_22.2: bool = converted %int.less_eq, %.loc36_22.1
// CHECK:STDOUT:   return %.loc36_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
