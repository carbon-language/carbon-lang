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

library "[[@TEST_NAME]]";

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

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(Eq(5, 5)) as Expect(true);
  Test(Eq(5, 6)) as Expect(false);
  Test(Eq(-1, -1)) as Expect(true);
  Test(Eq(-1, 1)) as Expect(false);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: i32) -> bool = "int.eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(Eq(5, 5)) as Expect(true);
  Test(Eq(5, 6)) as Expect(false);
  Test(Eq(-1, -1)) as Expect(true);
  Test(Eq(-1, 1)) as Expect(false);
}

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";

fn Test(n: Core.IntLiteral()) {
  // OK
  Eq(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(n, 1);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Eq(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(1, n);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Eq(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+6]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(n, n);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Eq(n, n);
}

// CHECK:STDOUT: --- int_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Eq.type.1: type = fn_type @Eq.1 [template]
// CHECK:STDOUT:   %Eq: %Eq.type.1 = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
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
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type.1 = fn_decl @Eq.1 [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_26.2: type = converted %bool.make_type, %.loc4_26.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_10: type = splice_block %i32.loc4_10 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_18: type = splice_block %i32.loc4_18 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %true_.param: %True = value_param runtime_param0
// CHECK:STDOUT:     %True.ref.loc9: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param runtime_param1
// CHECK:STDOUT:     %False.ref.loc9: type = name_ref False, file.%False.decl [template = constants.%False]
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
// CHECK:STDOUT:     %.loc14_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc14_35.2: type = converted %bool.make_type, %.loc14_35.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc14_19: type = splice_block %i32.loc14_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc14_27: type = splice_block %i32.loc14_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
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
// CHECK:STDOUT: fn @Eq.1(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Eq.ref.loc10: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc10_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc10_22: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc10_19: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_19: <bound method> = bound_method %int_1.loc10_19, %impl.elem0.loc10_19 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_19: <specific function> = specific_function %Convert.bound.loc10_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_19: init %i32 = call %Convert.specific_fn.loc10_19(%int_1.loc10_19) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.convert_checked.loc10_19 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int_1.loc10_19, %.loc10_19.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_22: <bound method> = bound_method %int_1.loc10_22, %impl.elem0.loc10_22 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_22: <specific function> = specific_function %Convert.bound.loc10_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_22: init %i32 = call %Convert.specific_fn.loc10_22(%int_1.loc10_22) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %int.convert_checked.loc10_22 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_1.loc10_22, %.loc10_22.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.eq.loc10: init bool = call %Eq.ref.loc10(%.loc10_19.2, %.loc10_22.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_13.1: bool = value_of_initializer %int.eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_13.2: bool = converted %int.eq.loc10, %.loc10_13.1 [template = constants.%true]
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
// CHECK:STDOUT:   %false_.ref: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Eq.ref.loc11: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc11_20: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_20: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_20 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_20: <specific function> = specific_function %Convert.bound.loc11_20, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_20: init %i32 = call %Convert.specific_fn.loc11_20(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_20.1: %i32 = value_of_initializer %int.convert_checked.loc11_20 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_20.2: %i32 = converted %int_1.loc11, %.loc11_20.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_23: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_23: <bound method> = bound_method %int_2, %impl.elem0.loc11_23 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_23: <specific function> = specific_function %Convert.bound.loc11_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_23: init %i32 = call %Convert.specific_fn.loc11_23(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_23.1: %i32 = value_of_initializer %int.convert_checked.loc11_23 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_23.2: %i32 = converted %int_2, %.loc11_23.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.eq.loc11: init bool = call %Eq.ref.loc11(%.loc11_20.2, %.loc11_23.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %int.eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %int.eq.loc11, %.loc11_14.1 [template = constants.%false]
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
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.eq: init bool = call %Eq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc15_18.1: bool = value_of_initializer %int.eq
// CHECK:STDOUT:   %.loc15_18.2: bool = converted %int.eq, %.loc15_18.1
// CHECK:STDOUT:   return %.loc15_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %WrongResult.type: type = fn_type @WrongResult [template]
// CHECK:STDOUT:   %WrongResult: %WrongResult.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
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
// CHECK:STDOUT:     %int_32.loc8_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_19: type = splice_block %i32.loc8_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc8_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc8_27: type = splice_block %i32.loc8_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc8_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongResult(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [template]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [template]
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
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type = fn_decl @Eq [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_54.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_54.2: type = converted %bool.make_type, %.loc4_54.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_26: init type = call %IntLiteral.ref.loc4_14() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %int_literal.make_type.loc4_26 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %int_literal.make_type.loc4_26, %.loc4_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_48.1: type = splice_block %.loc4_48.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_32: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_36: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_48: init type = call %IntLiteral.ref.loc4_36() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_48.2: type = value_of_initializer %int_literal.make_type.loc4_48 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_48.3: type = converted %int_literal.make_type.loc4_48, %.loc4_48.2 [template = Core.IntLiteral]
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
// CHECK:STDOUT: fn @Eq(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> bool = "int.eq";
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
// CHECK:STDOUT:   %Eq.ref.loc10: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_5.loc10_11: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_5.loc10_14: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int.eq.loc10: init bool = call %Eq.ref.loc10(%int_5.loc10_11, %int_5.loc10_14) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_16.1: bool = value_of_initializer %int.eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_16.2: bool = converted %int.eq.loc10, %.loc10_16.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_16.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_16.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_16.4: ref %Expect.2 = temporary %.loc10_16.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc11: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int.eq.loc11: init bool = call %Eq.ref.loc11(%int_5.loc11, %int_6) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_16.1: bool = value_of_initializer %int.eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_16.2: bool = converted %int.eq.loc11, %.loc11_16.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_16.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_16.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_16.4: ref %Expect.3 = temporary %.loc11_16.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc12: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc12_12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc12_11: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc12_11: <bound method> = bound_method %int_1.loc12_12, %impl.elem0.loc12_11 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc12_11: init Core.IntLiteral = call %Op.bound.loc12_11(%int_1.loc12_12) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc12_16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc12_15: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc12_15: <bound method> = bound_method %int_1.loc12_16, %impl.elem0.loc12_15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc12_15: init Core.IntLiteral = call %Op.bound.loc12_15(%int_1.loc12_16) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_11.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12_11 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_11.2: Core.IntLiteral = converted %int.snegate.loc12_11, %.loc12_11.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12_15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_15.2: Core.IntLiteral = converted %int.snegate.loc12_15, %.loc12_15.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.eq.loc12: init bool = call %Eq.ref.loc12(%.loc12_11.2, %.loc12_15.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_18.1: bool = value_of_initializer %int.eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_18.2: bool = converted %int.eq.loc12, %.loc12_18.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_18.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_18.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc12: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_18.4: ref %Expect.2 = temporary %.loc12_18.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc13: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc13_12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13: <bound method> = bound_method %int_1.loc13_12, %impl.elem0.loc13 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13: init Core.IntLiteral = call %Op.bound.loc13(%int_1.loc13_12) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc13_15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc13_11.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_11.2: Core.IntLiteral = converted %int.snegate.loc13, %.loc13_11.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.eq.loc13: init bool = call %Eq.ref.loc13(%.loc13_11.2, %int_1.loc13_15) [template = constants.%false]
// CHECK:STDOUT:   %.loc13_17.1: bool = value_of_initializer %int.eq.loc13 [template = constants.%false]
// CHECK:STDOUT:   %.loc13_17.2: bool = converted %int.eq.loc13, %.loc13_17.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc13_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.3 = call %Test.specific_fn.loc13() to %.loc13_17.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc13_17.4: ref %Expect.3 = temporary %.loc13_17.3, %Test.call.loc13
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
// CHECK:STDOUT:   %Eq.type.1: type = fn_type @Eq.1 [template]
// CHECK:STDOUT:   %Eq: %Eq.type.1 = struct_value () [template]
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
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type.1 = fn_decl @Eq.1 [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_40.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_40.2: type = converted %bool.make_type, %.loc4_40.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %int_literal.make_type, %.loc4_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_32: type = splice_block %i32 [template = constants.%i32] {
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
// CHECK:STDOUT: fn @Eq.1(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> bool = "int.eq";
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
// CHECK:STDOUT:   %Eq.ref.loc10: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_5.loc10_11: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_5.loc10_14: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int_5.loc10_14, %impl.elem0.loc10 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%int_5.loc10_14) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_14.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_14.2: %i32 = converted %int_5.loc10_14, %.loc10_14.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int.eq.loc10: init bool = call %Eq.ref.loc10(%int_5.loc10_11, %.loc10_14.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_16.1: bool = value_of_initializer %int.eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_16.2: bool = converted %int.eq.loc10, %.loc10_16.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_16.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_16.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_16.4: ref %Expect.2 = temporary %.loc10_16.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc11: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_6, %impl.elem0.loc11 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_6) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_14.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_14.2: %i32 = converted %int_6, %.loc11_14.1 [template = constants.%int_6.2]
// CHECK:STDOUT:   %int.eq.loc11: init bool = call %Eq.ref.loc11(%int_5.loc11, %.loc11_14.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_16.1: bool = value_of_initializer %int.eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_16.2: bool = converted %int.eq.loc11, %.loc11_16.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_16.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_16.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_16.4: ref %Expect.3 = temporary %.loc11_16.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc12: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc12_12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_11: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc12_11: <bound method> = bound_method %int_1.loc12_12, %impl.elem0.loc12_11 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc12_11: init Core.IntLiteral = call %Op.bound.loc12_11(%int_1.loc12_12) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc12_16: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_15.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc12_15: <bound method> = bound_method %int_1.loc12_16, %impl.elem0.loc12_15.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc12_15: init Core.IntLiteral = call %Op.bound.loc12_15(%int_1.loc12_16) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc12_11.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12_11 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc12_11.2: Core.IntLiteral = converted %int.snegate.loc12_11, %.loc12_11.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_15.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int.snegate.loc12_15, %impl.elem0.loc12_15.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc12_15.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12_15 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc12_15.2: Core.IntLiteral = converted %int.snegate.loc12_15, %.loc12_15.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%.loc12_15.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_15.3: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc12_15.4: %i32 = converted %int.snegate.loc12_15, %.loc12_15.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.eq.loc12: init bool = call %Eq.ref.loc12(%.loc12_11.2, %.loc12_15.4) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_18.1: bool = value_of_initializer %int.eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_18.2: bool = converted %int.eq.loc12, %.loc12_18.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_18.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_18.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc12: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_18.4: ref %Expect.2 = temporary %.loc12_18.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Eq.ref.loc13: %Eq.type.1 = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc13_12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_11: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc13: <bound method> = bound_method %int_1.loc13_12, %impl.elem0.loc13_11 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13: init Core.IntLiteral = call %Op.bound.loc13(%int_1.loc13_12) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc13_15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc13_11.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_11.2: Core.IntLiteral = converted %int.snegate.loc13, %.loc13_11.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_15: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int_1.loc13_15, %impl.elem0.loc13_15 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%int_1.loc13_15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_15.1: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_15.2: %i32 = converted %int_1.loc13_15, %.loc13_15.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.eq.loc13: init bool = call %Eq.ref.loc13(%.loc13_11.2, %.loc13_15.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc13_17.1: bool = value_of_initializer %int.eq.loc13 [template = constants.%false]
// CHECK:STDOUT:   %.loc13_17.2: bool = converted %int.eq.loc13, %.loc13_17.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc13_17.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.3 = call %Test.specific_fn.loc13() to %.loc13_17.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc13_17.4: ref %Expect.3 = temporary %.loc13_17.3, %Test.call.loc13
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
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [template]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [template]
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
// CHECK:STDOUT:     .Eq = %Eq.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Eq.decl: %Eq.type = fn_decl @Eq [template = constants.%Eq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_54.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_54.2: type = converted %bool.make_type, %.loc4_54.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_10: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_14: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_26: init type = call %IntLiteral.ref.loc4_14() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %int_literal.make_type.loc4_26 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %int_literal.make_type.loc4_26, %.loc4_26.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_48.1: type = splice_block %.loc4_48.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_32: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_36: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_48: init type = call %IntLiteral.ref.loc4_36() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_48.2: type = value_of_initializer %int_literal.make_type.loc4_48 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_48.3: type = converted %int_literal.make_type.loc4_48, %.loc4_48.2 [template = Core.IntLiteral]
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
// CHECK:STDOUT: fn @Eq(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> bool = "int.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%n.param_patt: Core.IntLiteral) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref.loc8: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc8_6: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc8_9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.eq.loc8: init bool = call %Eq.ref.loc8(%int_1.loc8_6, %int_1.loc8_9) [template = constants.%true]
// CHECK:STDOUT:   %Eq.ref.loc16: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %n.ref.loc16: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.eq.loc16: init bool = call %Eq.ref.loc16(%n.ref.loc16, %int_1.loc16)
// CHECK:STDOUT:   %Eq.ref.loc24: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %n.ref.loc24: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.eq.loc24: init bool = call %Eq.ref.loc24(%int_1.loc24, %n.ref.loc24)
// CHECK:STDOUT:   %Eq.ref.loc31: %Eq.type = name_ref Eq, file.%Eq.decl [template = constants.%Eq]
// CHECK:STDOUT:   %n.ref.loc31_6: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc31_9: Core.IntLiteral = name_ref n, %n
// CHECK:STDOUT:   %int.eq.loc31: init bool = call %Eq.ref.loc31(%n.ref.loc31_6, %n.ref.loc31_9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
