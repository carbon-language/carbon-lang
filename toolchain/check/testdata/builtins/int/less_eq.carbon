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

library "[[@TEST_NAME]]";

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

// --- literal.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(LessEq(5, 5)) as Expect(true);
  Test(LessEq(5, 6)) as Expect(true);
  Test(LessEq(6, 5)) as Expect(false);
  Test(LessEq(-1, -1)) as Expect(true);
  Test(LessEq(-1, 1)) as Expect(true);
  Test(LessEq(1, -1)) as Expect(false);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: i32) -> bool = "int.less_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(LessEq(5, 5)) as Expect(true);
  Test(LessEq(5, 6)) as Expect(true);
  Test(LessEq(6, 5)) as Expect(false);
  Test(LessEq(-1, -1)) as Expect(true);
  Test(LessEq(-1, 1)) as Expect(true);
  Test(LessEq(1, -1)) as Expect(false);
}

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";

fn Test(n: Core.IntLiteral()) {
  // OK
  LessEq(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(n, 1);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  LessEq(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(1, n);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  LessEq(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+6]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(n, n);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  LessEq(n, n);
}

// CHECK:STDOUT: --- int_less_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [template]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.1, %Convert.10 [template]
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
// CHECK:STDOUT:     .Bool = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
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
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_30.2: type = converted %bool.make_type, %.loc4_30.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_14: type = splice_block %i32.loc4_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_22: type = splice_block %i32.loc4_22 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %true_.param: %True = value_param runtime_param0
// CHECK:STDOUT:     %True.ref.loc10: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param runtime_param1
// CHECK:STDOUT:     %False.ref.loc10: type = name_ref False, file.%False.decl [template = constants.%False]
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
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_35.2: type = converted %bool.make_type, %.loc18_35.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_19: type = splice_block %i32.loc18_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc18_27: type = splice_block %i32.loc18_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessEq(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.less_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc11: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc11: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc11_23: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_23: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_23: <specific function> = specific_function %Convert.bound.loc11_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_23: init %i32 = call %Convert.specific_fn.loc11_23(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_23.1: %i32 = value_of_initializer %int.convert_checked.loc11_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_23.2: %i32 = converted %int_1.loc11, %.loc11_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_26: <bound method> = bound_method %int_2, %impl.elem0.loc11_26 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_26: <specific function> = specific_function %Convert.bound.loc11_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_26: init %i32 = call %Convert.specific_fn.loc11_26(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_26.1: %i32 = value_of_initializer %int.convert_checked.loc11_26 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_26.2: %i32 = converted %int_2, %.loc11_26.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.less_eq.loc11: init bool = call %LessEq.ref.loc11(%.loc11_23.2, %.loc11_26.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc11_13.1: bool = value_of_initializer %int.less_eq.loc11 [template = constants.%true]
// CHECK:STDOUT:   %.loc11_13.2: bool = converted %int.less_eq.loc11, %.loc11_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc11_13.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
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
// CHECK:STDOUT:   %.loc11_13.3: type = block_arg !if.expr.result.loc11 [template = constants.%True]
// CHECK:STDOUT:   %true_.ref.loc12: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc12: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc12_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc12_26: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_23: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_23: <bound method> = bound_method %int_1.loc12_23, %impl.elem0.loc12_23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_23: <specific function> = specific_function %Convert.bound.loc12_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_23: init %i32 = call %Convert.specific_fn.loc12_23(%int_1.loc12_23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_23.1: %i32 = value_of_initializer %int.convert_checked.loc12_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_23.2: %i32 = converted %int_1.loc12_23, %.loc12_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_26: <bound method> = bound_method %int_1.loc12_26, %impl.elem0.loc12_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_26: <specific function> = specific_function %Convert.bound.loc12_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_26: init %i32 = call %Convert.specific_fn.loc12_26(%int_1.loc12_26) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_26.1: %i32 = value_of_initializer %int.convert_checked.loc12_26 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_26.2: %i32 = converted %int_1.loc12_26, %.loc12_26.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.less_eq.loc12: init bool = call %LessEq.ref.loc12(%.loc12_23.2, %.loc12_26.2) [template = constants.%true]
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
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc13_24: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_24: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_24 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_24: <specific function> = specific_function %Convert.bound.loc13_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_24: init %i32 = call %Convert.specific_fn.loc13_24(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_24.1: %i32 = value_of_initializer %int.convert_checked.loc13_24 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_24.2: %i32 = converted %int_1.loc13, %.loc13_24.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_27: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_27: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13_27 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_27: <specific function> = specific_function %Convert.bound.loc13_27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_27: init %i32 = call %Convert.specific_fn.loc13_27(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_27.1: %i32 = value_of_initializer %int.convert_checked.loc13_27 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_27.2: %i32 = converted %int_0.loc13, %.loc13_27.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.less_eq.loc13: init bool = call %LessEq.ref.loc13(%.loc13_24.2, %.loc13_27.2) [template = constants.%false]
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
// CHECK:STDOUT:   %true_.ref.loc14: %True = name_ref true_, %true_
// CHECK:STDOUT:   %LessEq.ref.loc14: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %Negate.ref.loc14: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_30: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_30: <specific function> = specific_function %Convert.bound.loc14_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_30: init %i32 = call %Convert.specific_fn.loc14_30(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_30.1: %i32 = value_of_initializer %int.convert_checked.loc14_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_30.2: %i32 = converted %int_1.loc14, %.loc14_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc14: init %i32 = call %Negate.ref.loc14(%.loc14_30.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_0.loc14: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc14_31.1: %i32 = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc14_31.2: %i32 = converted %int.snegate.loc14, %.loc14_31.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_34: <bound method> = bound_method %int_0.loc14, %impl.elem0.loc14_34 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_34: <specific function> = specific_function %Convert.bound.loc14_34, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_34: init %i32 = call %Convert.specific_fn.loc14_34(%int_0.loc14) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.convert_checked.loc14_34 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_0.loc14, %.loc14_34.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.less_eq.loc14: init bool = call %LessEq.ref.loc14(%.loc14_31.2, %.loc14_34.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc14_13.1: bool = value_of_initializer %int.less_eq.loc14 [template = constants.%true]
// CHECK:STDOUT:   %.loc14_13.2: bool = converted %int.less_eq.loc14, %.loc14_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc14_13.2 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %True.ref.loc14: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc14(%True.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %False.ref.loc14: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc14(%False.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_13.3: type = block_arg !if.expr.result.loc14 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc15: %False = name_ref false_, %false_
// CHECK:STDOUT:   %LessEq.ref.loc15: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_0.loc15: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %Negate.ref.loc15: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_34: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_34: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_34 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_34: <specific function> = specific_function %Convert.bound.loc15_34, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_34: init %i32 = call %Convert.specific_fn.loc15_34(%int_1.loc15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_34.1: %i32 = value_of_initializer %int.convert_checked.loc15_34 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_34.2: %i32 = converted %int_1.loc15, %.loc15_34.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc15: init %i32 = call %Negate.ref.loc15(%.loc15_34.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc15_24: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_24: <bound method> = bound_method %int_0.loc15, %impl.elem0.loc15_24 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_24: <specific function> = specific_function %Convert.bound.loc15_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_24: init %i32 = call %Convert.specific_fn.loc15_24(%int_0.loc15) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_24.1: %i32 = value_of_initializer %int.convert_checked.loc15_24 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_24.2: %i32 = converted %int_0.loc15, %.loc15_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_35.1: %i32 = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_35.2: %i32 = converted %int.snegate.loc15, %.loc15_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.less_eq.loc15: init bool = call %LessEq.ref.loc15(%.loc15_24.2, %.loc15_35.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc15_14.1: bool = value_of_initializer %int.less_eq.loc15 [template = constants.%false]
// CHECK:STDOUT:   %.loc15_14.2: bool = converted %int.less_eq.loc15, %.loc15_14.1 [template = constants.%false]
// CHECK:STDOUT:   if %.loc15_14.2 br !if.expr.then.loc15 else br !if.expr.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc15:
// CHECK:STDOUT:   %True.ref.loc15: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc15(%True.ref.loc15)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc15:
// CHECK:STDOUT:   %False.ref.loc15: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc15(%False.ref.loc15)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc15:
// CHECK:STDOUT:   %.loc15_14.3: type = block_arg !if.expr.result.loc15 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LessEq.ref: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.less_eq: init bool = call %LessEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc19_22.1: bool = value_of_initializer %int.less_eq
// CHECK:STDOUT:   %.loc19_22.2: bool = converted %int.less_eq, %.loc19_22.1
// CHECK:STDOUT:   return %.loc19_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [template]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%true) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%true) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%false) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%false) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .Negate = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LessEq = %LessEq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LessEq.decl: %LessEq.type = fn_decl @LessEq [template = constants.%LessEq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_58.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_58.2: type = converted %bool.make_type, %.loc4_58.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %int_literal.make_type.loc4_30, %.loc4_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_52.1: type = splice_block %.loc4_52.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.2: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.3: type = converted %int_literal.make_type.loc4_52, %.loc4_52.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %B.patt.loc6_14.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc6_14.1, runtime_param<invalid> [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc6_18.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc6_18.3: type = converted %bool.make_type, %.loc6_18.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_14.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %B.patt.loc7_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc7_9.1, runtime_param<invalid> [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %B.ref: bool = name_ref B, %B.loc7_9.1 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %Expect.loc7_30.1: type = class_type @Expect, @Expect(constants.%B) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc7_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc7_13.3: type = converted %bool.make_type, %.loc7_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc7_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_30.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%B.loc6_14.1: bool) {
// CHECK:STDOUT:   %B.loc6_14.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_14.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessEq(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> bool = "int.less_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%B.loc7_9.1: bool) {
// CHECK:STDOUT:   %B.loc7_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc7_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   %Expect.loc7_30.2: type = class_type @Expect, @Expect(%B.loc7_9.2) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_30.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_30.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%B.param_patt: bool) -> %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_42.2: init @Test.%Expect.loc7_30.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_43: init @Test.%Expect.loc7_30.2 (%Expect.1) = converted %.loc7_42.1, %.loc7_42.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_43 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc10: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_5.loc10_15: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_5.loc10_18: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int.less_eq.loc10: init bool = call %LessEq.ref.loc10(%int_5.loc10_15, %int_5.loc10_18) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_20.1: bool = value_of_initializer %int.less_eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_20.2: bool = converted %int.less_eq.loc10, %.loc10_20.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_20.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_20.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_20.4: ref %Expect.2 = temporary %.loc10_20.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc11: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_6.loc11: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int.less_eq.loc11: init bool = call %LessEq.ref.loc11(%int_5.loc11, %int_6.loc11) [template = constants.%true]
// CHECK:STDOUT:   %.loc11_20.1: bool = value_of_initializer %int.less_eq.loc11 [template = constants.%true]
// CHECK:STDOUT:   %.loc11_20.2: bool = converted %int.less_eq.loc11, %.loc11_20.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_20.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_20.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc11: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_20.4: ref %Expect.2 = temporary %.loc11_20.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc12: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_6.loc12: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int_5.loc12: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int.less_eq.loc12: init bool = call %LessEq.ref.loc12(%int_6.loc12, %int_5.loc12) [template = constants.%false]
// CHECK:STDOUT:   %.loc12_20.1: bool = value_of_initializer %int.less_eq.loc12 [template = constants.%false]
// CHECK:STDOUT:   %.loc12_20.2: bool = converted %int.less_eq.loc12, %.loc12_20.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_20.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_20.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_20.4: ref %Expect.3 = temporary %.loc12_20.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc13: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc13_16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13_15: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13_15: <bound method> = bound_method %int_1.loc13_16, %impl.elem0.loc13_15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_15: init Core.IntLiteral = call %Op.bound.loc13_15(%int_1.loc13_16) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc13_20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13_19: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13_19: <bound method> = bound_method %int_1.loc13_20, %impl.elem0.loc13_19 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_19: init Core.IntLiteral = call %Op.bound.loc13_19(%int_1.loc13_20) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_15.2: Core.IntLiteral = converted %int.snegate.loc13_15, %.loc13_15.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_19 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_19.2: Core.IntLiteral = converted %int.snegate.loc13_19, %.loc13_19.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.less_eq.loc13: init bool = call %LessEq.ref.loc13(%.loc13_15.2, %.loc13_19.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc13_22.1: bool = value_of_initializer %int.less_eq.loc13 [template = constants.%true]
// CHECK:STDOUT:   %.loc13_22.2: bool = converted %int.less_eq.loc13, %.loc13_22.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_22.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_22.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc13: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_22.4: ref %Expect.2 = temporary %.loc13_22.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc14: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc14_16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc14: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %int_1.loc14_16, %impl.elem0.loc14 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc14: init Core.IntLiteral = call %Op.bound.loc14(%int_1.loc14_16) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc14_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc14_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc14_15.2: Core.IntLiteral = converted %int.snegate.loc14, %.loc14_15.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.less_eq.loc14: init bool = call %LessEq.ref.loc14(%.loc14_15.2, %int_1.loc14_19) [template = constants.%true]
// CHECK:STDOUT:   %.loc14_21.1: bool = value_of_initializer %int.less_eq.loc14 [template = constants.%true]
// CHECK:STDOUT:   %.loc14_21.2: bool = converted %int.less_eq.loc14, %.loc14_21.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc14_21.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.2 = call %Test.specific_fn.loc14() to %.loc14_21.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc14: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc14_21.4: ref %Expect.2 = temporary %.loc14_21.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc15: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc15_15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc15_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int_1.loc15_19, %impl.elem0.loc15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc15: init Core.IntLiteral = call %Op.bound.loc15(%int_1.loc15_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_18.2: Core.IntLiteral = converted %int.snegate.loc15, %.loc15_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.less_eq.loc15: init bool = call %LessEq.ref.loc15(%int_1.loc15_15, %.loc15_18.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc15_21.1: bool = value_of_initializer %int.less_eq.loc15 [template = constants.%false]
// CHECK:STDOUT:   %.loc15_21.2: bool = converted %int.less_eq.loc15, %.loc15_21.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc15_21.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.3 = call %Test.specific_fn.loc15() to %.loc15_21.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc15: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc15_21.4: ref %Expect.3 = temporary %.loc15_21.3, %Test.call.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%B) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%B.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%B) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%true) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%true) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%false) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%false) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [template]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%true) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%true) [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_6.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%false) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%false) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.6
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     .Negate = %import_ref.231
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LessEq = %LessEq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LessEq.decl: %LessEq.type = fn_decl @LessEq [template = constants.%LessEq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_44.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_44.2: type = converted %bool.make_type, %.loc4_44.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %int_literal.make_type, %.loc4_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_36: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %B.patt.loc6_14.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc6_14.1, runtime_param<invalid> [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc6_18.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc6_18.3: type = converted %bool.make_type, %.loc6_18.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_14.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %B.patt.loc7_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc7_9.1, runtime_param<invalid> [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %B.ref: bool = name_ref B, %B.loc7_9.1 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %Expect.loc7_30.1: type = class_type @Expect, @Expect(constants.%B) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc7_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc7_13.3: type = converted %bool.make_type, %.loc7_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc7_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_30.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%B.loc6_14.1: bool) {
// CHECK:STDOUT:   %B.loc6_14.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_14.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessEq(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> bool = "int.less_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%B.loc7_9.1: bool) {
// CHECK:STDOUT:   %B.loc7_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc7_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   %Expect.loc7_30.2: type = class_type @Expect, @Expect(%B.loc7_9.2) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_30.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_30.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%B.param_patt: bool) -> %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_42.2: init @Test.%Expect.loc7_30.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_43: init @Test.%Expect.loc7_30.2 (%Expect.1) = converted %.loc7_42.1, %.loc7_42.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_43 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc10: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_5.loc10_15: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_5.loc10_18: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int_5.loc10_18, %impl.elem0.loc10 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%int_5.loc10_18) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_18.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_18.2: %i32 = converted %int_5.loc10_18, %.loc10_18.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int.less_eq.loc10: init bool = call %LessEq.ref.loc10(%int_5.loc10_15, %.loc10_18.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_20.1: bool = value_of_initializer %int.less_eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_20.2: bool = converted %int.less_eq.loc10, %.loc10_20.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_20.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_20.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_20.4: ref %Expect.2 = temporary %.loc10_20.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc11: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_6.loc11: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_6.loc11, %impl.elem0.loc11 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_6.loc11) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_18.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_18.2: %i32 = converted %int_6.loc11, %.loc11_18.1 [template = constants.%int_6.2]
// CHECK:STDOUT:   %int.less_eq.loc11: init bool = call %LessEq.ref.loc11(%int_5.loc11, %.loc11_18.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc11_20.1: bool = value_of_initializer %int.less_eq.loc11 [template = constants.%true]
// CHECK:STDOUT:   %.loc11_20.2: bool = converted %int.less_eq.loc11, %.loc11_20.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_20.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_20.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc11: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_20.4: ref %Expect.2 = temporary %.loc11_20.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc12: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_6.loc12: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %int_5.loc12: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_5.loc12, %impl.elem0.loc12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_5.loc12) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc12_18.1: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc12_18.2: %i32 = converted %int_5.loc12, %.loc12_18.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int.less_eq.loc12: init bool = call %LessEq.ref.loc12(%int_6.loc12, %.loc12_18.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc12_20.1: bool = value_of_initializer %int.less_eq.loc12 [template = constants.%false]
// CHECK:STDOUT:   %.loc12_20.2: bool = converted %int.less_eq.loc12, %.loc12_20.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc12_20.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.3 = call %Test.specific_fn.loc12() to %.loc12_20.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc12_20.4: ref %Expect.3 = temporary %.loc12_20.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc13: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc13_16: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_15: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc13_15: <bound method> = bound_method %int_1.loc13_16, %impl.elem0.loc13_15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_15: init Core.IntLiteral = call %Op.bound.loc13_15(%int_1.loc13_16) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc13_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_19.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc13_19: <bound method> = bound_method %int_1.loc13_20, %impl.elem0.loc13_19.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_19: init Core.IntLiteral = call %Op.bound.loc13_19(%int_1.loc13_20) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_15 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_15.2: Core.IntLiteral = converted %int.snegate.loc13_15, %.loc13_15.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_19.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int.snegate.loc13_19, %impl.elem0.loc13_19.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc13_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_19 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_19.2: Core.IntLiteral = converted %int.snegate.loc13_19, %.loc13_19.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%.loc13_19.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc13_19.3: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc13_19.4: %i32 = converted %int.snegate.loc13_19, %.loc13_19.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.less_eq.loc13: init bool = call %LessEq.ref.loc13(%.loc13_15.2, %.loc13_19.4) [template = constants.%true]
// CHECK:STDOUT:   %.loc13_22.1: bool = value_of_initializer %int.less_eq.loc13 [template = constants.%true]
// CHECK:STDOUT:   %.loc13_22.2: bool = converted %int.less_eq.loc13, %.loc13_22.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_22.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_22.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc13: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_22.4: ref %Expect.2 = temporary %.loc13_22.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc14: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc14_16: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_15: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %int_1.loc14_16, %impl.elem0.loc14_15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc14: init Core.IntLiteral = call %Op.bound.loc14(%int_1.loc14_16) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc14_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc14_15.2: Core.IntLiteral = converted %int.snegate.loc14, %.loc14_15.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_19: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int_1.loc14_19, %impl.elem0.loc14_19 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.specific_fn.loc14(%int_1.loc14_19) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_19.1: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_19.2: %i32 = converted %int_1.loc14_19, %.loc14_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.less_eq.loc14: init bool = call %LessEq.ref.loc14(%.loc14_15.2, %.loc14_19.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc14_21.1: bool = value_of_initializer %int.less_eq.loc14 [template = constants.%true]
// CHECK:STDOUT:   %.loc14_21.2: bool = converted %int.less_eq.loc14, %.loc14_21.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc14_21.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.2 = call %Test.specific_fn.loc14() to %.loc14_21.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc14: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc14_21.4: ref %Expect.2 = temporary %.loc14_21.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LessEq.ref.loc15: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc15_15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc15_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_18.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int_1.loc15_19, %impl.elem0.loc15_18.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc15: init Core.IntLiteral = call %Op.bound.loc15(%int_1.loc15_19) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_18.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int.snegate.loc15, %impl.elem0.loc15_18.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc15_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_18.2: Core.IntLiteral = converted %int.snegate.loc15, %.loc15_18.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%.loc15_18.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_18.3: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_18.4: %i32 = converted %int.snegate.loc15, %.loc15_18.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.less_eq.loc15: init bool = call %LessEq.ref.loc15(%int_1.loc15_15, %.loc15_18.4) [template = constants.%false]
// CHECK:STDOUT:   %.loc15_21.1: bool = value_of_initializer %int.less_eq.loc15 [template = constants.%false]
// CHECK:STDOUT:   %.loc15_21.2: bool = converted %int.less_eq.loc15, %.loc15_21.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc15_21.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.3 = call %Test.specific_fn.loc15() to %.loc15_21.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc15: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc15_21.4: ref %Expect.3 = temporary %.loc15_21.3, %Test.call.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%B) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%B.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%B) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%true) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%true) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%false) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%false) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_runtime_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %LessEq.type: type = fn_type @LessEq [template]
// CHECK:STDOUT:   %LessEq: %LessEq.type = struct_value () [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LessEq = %LessEq.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LessEq.decl: %LessEq.type = fn_decl @LessEq [template = constants.%LessEq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_58.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_58.2: type = converted %bool.make_type, %.loc4_58.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %int_literal.make_type.loc4_30, %.loc4_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_52.1: type = splice_block %.loc4_52.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.2: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_52.3: type = converted %int_literal.make_type.loc4_52, %.loc4_52.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %n.patt: Core.IntLiteral = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: Core.IntLiteral = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc6_28.1: type = splice_block %.loc6_28.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_28.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_28.3: type = converted %int_literal.make_type, %.loc6_28.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: Core.IntLiteral = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LessEq(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> bool = "int.less_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%n.param_patt: Core.IntLiteral) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LessEq.ref.loc8: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc8_10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc8_13: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.less_eq.loc8: init bool = call %LessEq.ref.loc8(%int_1.loc8_10, %int_1.loc8_13) [template = constants.%true]
// CHECK:STDOUT:   %LessEq.ref.loc16: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %n.ref.loc16: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.less_eq.loc16: init bool = call %LessEq.ref.loc16(%n.ref.loc16, %int_1.loc16)
// CHECK:STDOUT:   %LessEq.ref.loc24: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %n.ref.loc24: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.less_eq.loc24: init bool = call %LessEq.ref.loc24(%int_1.loc24, %n.ref.loc24)
// CHECK:STDOUT:   %LessEq.ref.loc31: %LessEq.type = name_ref LessEq, file.%LessEq.decl [template = constants.%LessEq]
// CHECK:STDOUT:   %n.ref.loc31_10: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc31_13: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.less_eq.loc31: init bool = call %LessEq.ref.loc31(%n.ref.loc31_10, %n.ref.loc31_13)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
