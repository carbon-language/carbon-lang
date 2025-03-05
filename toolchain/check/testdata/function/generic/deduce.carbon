// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/deduce.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/deduce.carbon

// --- deduce_explicit.carbon

library "[[@TEST_NAME]]";

fn ExplicitGenericParam(T:! type) -> T* { return ExplicitGenericParam(T); }

fn CallExplicitGenericParam() -> i32* {
  return ExplicitGenericParam(i32);
}

fn CallExplicitGenericParamWithGenericArg(T:! type) -> {.a: T}* {
  return ExplicitGenericParam({.a: T});
}

// --- fail_deduce_explicit_non_constant.carbon

library "[[@TEST_NAME]]";

fn ExplicitGenericParam(T:! type) -> T* { return ExplicitGenericParam(T); }

fn CallExplicitGenericParamConst(T:! type) {
  ExplicitGenericParam(T);
}

fn CallExplicitGenericParamNonConst(T: type) {
  // CHECK:STDERR: fail_deduce_explicit_non_constant.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   ExplicitGenericParam(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_explicit_non_constant.carbon:[[@LINE-10]]:25: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn ExplicitGenericParam(T:! type) -> T* { return ExplicitGenericParam(T); }
  // CHECK:STDERR:                         ^
  // CHECK:STDERR:
  ExplicitGenericParam(T);
}

// --- explicit_vs_deduced.carbon

library "[[@TEST_NAME]]";

class A {}

fn ExplicitAndAlsoDeduced(T:! type, x: T) -> T* {
  return ExplicitAndAlsoDeduced(T, x);
}

fn CallExplicitAndAlsoDeduced() -> A* {
  return ExplicitAndAlsoDeduced(A, {});
}

// --- deduce_implicit.carbon

library "[[@TEST_NAME]]";

fn ImplicitGenericParam[T:! type](x: T) -> T* { return ImplicitGenericParam(x); }

fn CallImplicitGenericParam(n: i32) -> i32* {
  return ImplicitGenericParam(n);
}

// --- deduce_nested_tuple.carbon

library "[[@TEST_NAME]]";

fn TupleParam[T:! type](x: (T, i32)) {}

fn CallTupleParam() {
  TupleParam((1, 2));
}

// --- deduce_nested_struct.carbon

library "[[@TEST_NAME]]";

fn StructParam[T:! type](x: {.a: T, .b: i32}) {}

fn CallStructParam() {
  StructParam({.a = 1, .b = 2});
}

// --- fail_deduce_bigger_struct.carbon

library "[[@TEST_NAME]]";

fn BigStructParam[T:! type](x: {.c: T, .d: i32, .e: i32}) {}

fn CallBigStructParam() {
  // CHECK:STDERR: fail_deduce_bigger_struct.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   BigStructParam({.c = 3, .d = 4});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_bigger_struct.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn BigStructParam[T:! type](x: {.c: T, .d: i32, .e: i32}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  BigStructParam({.c = 3, .d = 4});
}

// --- fail_deduce_smaller_struct.carbon

library "[[@TEST_NAME]]";

fn SmallStructParam[T:! type](x: {.f: T, .g: i32}) {}

fn CallSmallStructParam() {
  // CHECK:STDERR: fail_deduce_smaller_struct.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   SmallStructParam({.f = 5, .g = 6, .h = 7});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_smaller_struct.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn SmallStructParam[T:! type](x: {.f: T, .g: i32}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  SmallStructParam({.f = 5, .g = 6, .h = 7});
}

// --- fail_deduce_struct_wrong_name.carbon

library "[[@TEST_NAME]]";

fn WrongNameStructParam[T:! type](x: {.i: T, .different: i32}) {}

fn CallWrongNameStructParam() {
  // CHECK:STDERR: fail_deduce_struct_wrong_name.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   WrongNameStructParam({.i = 8, .j = 9});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_struct_wrong_name.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn WrongNameStructParam[T:! type](x: {.i: T, .different: i32}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WrongNameStructParam({.i = 8, .j = 9});
}

// --- fail_todo_deduce_struct_wrong_order.carbon

library "[[@TEST_NAME]]";

fn WrongOrderStructParam[T:! type](x: {.first: T, .second: i32}) {}

fn CallWrongOrderStructParam() {
  // CHECK:STDERR: fail_todo_deduce_struct_wrong_order.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   WrongOrderStructParam({.second = 11, .first = 10});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_struct_wrong_order.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn WrongOrderStructParam[T:! type](x: {.first: T, .second: i32}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  WrongOrderStructParam({.second = 11, .first = 10});
}

// --- fail_deduce_incomplete.carbon

library "[[@TEST_NAME]]";

// TODO: It would be nice to diagnose this at its point of declaration, because
// U is not deducible.
fn ImplicitNotDeducible[T:! type, U:! type](x: T) -> U;

fn CallImplicitNotDeducible() {
  // CHECK:STDERR: fail_deduce_incomplete.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `U` [DeductionIncomplete]
  // CHECK:STDERR:   ImplicitNotDeducible(42);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_incomplete.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn ImplicitNotDeducible[T:! type, U:! type](x: T) -> U;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ImplicitNotDeducible(42);
}

// --- fail_deduce_inconsistent.carbon

library "[[@TEST_NAME]]";

fn ImplicitNotDeducible[T:! type](x: T, y: T) -> T;

fn CallImplicitNotDeducible() {
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   ImplicitNotDeducible(42, {.x = 12});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn ImplicitNotDeducible[T:! type](x: T, y: T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ImplicitNotDeducible(42, {.x = 12});
}

// CHECK:STDOUT: --- deduce_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [concrete]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.c0a: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParam.type: type = fn_type @CallExplicitGenericParam [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParam: %CallExplicitGenericParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.74e: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%i32) [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %ptr.48a: type = ptr_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.type: type = fn_type @CallExplicitGenericParamWithGenericArg [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg: %CallExplicitGenericParamWithGenericArg.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.86d: <witness> = require_complete_type %ptr.48a [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.6ad: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%struct_type.a) [symbolic]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ExplicitGenericParam = %ExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParam = %CallExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParamWithGenericArg = %CallExplicitGenericParamWithGenericArg.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ExplicitGenericParam.decl: %ExplicitGenericParam.type = fn_decl @ExplicitGenericParam [concrete = constants.%ExplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_39.1: type = ptr_type %T [symbolic = %ptr.loc4_39.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParam.decl: %CallExplicitGenericParam.type = fn_decl @CallExplicitGenericParam [concrete = constants.%CallExplicitGenericParam] {
// CHECK:STDOUT:     %return.patt: %ptr.235 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.235 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.decl: %CallExplicitGenericParamWithGenericArg.type = fn_decl @CallExplicitGenericParamWithGenericArg [concrete = constants.%CallExplicitGenericParamWithGenericArg] {
// CHECK:STDOUT:     %T.patt.loc10_43.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %struct_type.a.loc10_62.1: type = struct_type {.a: %T} [symbolic = %struct_type.a.loc10_62.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %ptr.loc10_63.1: type = ptr_type %struct_type.a [symbolic = %ptr.loc10_63.2 (constants.%ptr.48a)]
// CHECK:STDOUT:     %T.loc10_43.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = out_param call_param0
// CHECK:STDOUT:     %return: ref @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %ptr.loc4_39.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) [symbolic = %require_complete (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %ExplicitGenericParam.specific_fn.loc4_50.2 (constants.%ExplicitGenericParam.specific_fn.c0a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_25.1: type) -> @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %ExplicitGenericParam.specific_fn.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %ExplicitGenericParam.specific_fn.loc4_50.2 (constants.%ExplicitGenericParam.specific_fn.c0a)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = call %ExplicitGenericParam.specific_fn.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr.79f) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParam() -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%i32) [concrete = constants.%ExplicitGenericParam.specific_fn.74e]
// CHECK:STDOUT:   %ExplicitGenericParam.call: init %ptr.235 = call %ExplicitGenericParam.specific_fn()
// CHECK:STDOUT:   %.loc7_35.1: %ptr.235 = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_35.2: %ptr.235 = converted %ExplicitGenericParam.call, %.loc7_35.1
// CHECK:STDOUT:   return %.loc7_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamWithGenericArg(%T.loc10_43.1: type) {
// CHECK:STDOUT:   %T.loc10_43.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_43.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.a.loc10_62.2: type = struct_type {.a: @CallExplicitGenericParamWithGenericArg.%T.loc10_43.2 (%T)} [symbolic = %struct_type.a.loc10_62.2 (constants.%struct_type.a)]
// CHECK:STDOUT:   %ptr.loc10_63.2: type = ptr_type @CallExplicitGenericParamWithGenericArg.%struct_type.a.loc10_62.2 (%struct_type.a) [symbolic = %ptr.loc10_63.2 (constants.%ptr.48a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) [symbolic = %require_complete (constants.%require_complete.86d)]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc11_10.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%struct_type.a.loc10_62.2) [symbolic = %ExplicitGenericParam.specific_fn.loc11_10.2 (constants.%ExplicitGenericParam.specific_fn.6ad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc10_43.1: type) -> @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc11: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %struct_type.a.loc11: type = struct_type {.a: %T} [symbolic = %struct_type.a.loc10_62.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     %ExplicitGenericParam.specific_fn.loc11_10.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%struct_type.a) [symbolic = %ExplicitGenericParam.specific_fn.loc11_10.2 (constants.%ExplicitGenericParam.specific_fn.6ad)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = call %ExplicitGenericParam.specific_fn.loc11_10.1()
// CHECK:STDOUT:     %.loc11_39.1: @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc11_39.2: @CallExplicitGenericParamWithGenericArg.%ptr.loc10_63.2 (%ptr.48a) = converted %ExplicitGenericParam.call, %.loc11_39.1
// CHECK:STDOUT:     return %.loc11_39.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc4_39.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2 => constants.%ExplicitGenericParam.specific_fn.c0a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(%T.loc4_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %ptr.loc4_39.2 => constants.%ptr.235
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3d0
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2 => constants.%ExplicitGenericParam.specific_fn.74e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamWithGenericArg(constants.%T) {
// CHECK:STDOUT:   %T.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %struct_type.a.loc10_62.2 => constants.%struct_type.a
// CHECK:STDOUT:   %ptr.loc10_63.2 => constants.%ptr.48a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%struct_type.a) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%struct_type.a
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%struct_type.a
// CHECK:STDOUT:   %ptr.loc4_39.2 => constants.%ptr.48a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.86d
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2 => constants.%ExplicitGenericParam.specific_fn.6ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(@CallExplicitGenericParamWithGenericArg.%struct_type.a.loc10_62.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_explicit_non_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [concrete]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamConst.type: type = fn_type @CallExplicitGenericParamConst [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParamConst: %CallExplicitGenericParamConst.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst.type: type = fn_type @CallExplicitGenericParamNonConst [concrete]
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst: %CallExplicitGenericParamNonConst.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ExplicitGenericParam = %ExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParamConst = %CallExplicitGenericParamConst.decl
// CHECK:STDOUT:     .CallExplicitGenericParamNonConst = %CallExplicitGenericParamNonConst.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ExplicitGenericParam.decl: %ExplicitGenericParam.type = fn_decl @ExplicitGenericParam [concrete = constants.%ExplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_39.1: type = ptr_type %T [symbolic = %ptr.loc4_39.2 (constants.%ptr)]
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return: ref @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamConst.decl: %CallExplicitGenericParamConst.type = fn_decl @CallExplicitGenericParamConst [concrete = constants.%CallExplicitGenericParamConst] {
// CHECK:STDOUT:     %T.patt.loc6_34.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_34.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst.decl: %CallExplicitGenericParamNonConst.type = fn_decl @CallExplicitGenericParamNonConst [concrete = constants.%CallExplicitGenericParamNonConst] {
// CHECK:STDOUT:     %T.patt: type = binding_pattern T
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param call_param0
// CHECK:STDOUT:     %T: type = bind_name T, %T.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %ptr.loc4_39.2 (constants.%ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %ExplicitGenericParam.specific_fn.loc4_50.2 (constants.%ExplicitGenericParam.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_25.1: type) -> @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %ExplicitGenericParam.specific_fn.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %ExplicitGenericParam.specific_fn.loc4_50.2 (constants.%ExplicitGenericParam.specific_fn)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = call %ExplicitGenericParam.specific_fn.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%ptr.loc4_39.2 (%ptr) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamConst(%T.loc6_34.1: type) {
// CHECK:STDOUT:   %T.loc6_34.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_34.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc7_3.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc6_34.2) [symbolic = %ExplicitGenericParam.specific_fn.loc7_3.2 (constants.%ExplicitGenericParam.specific_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type @CallExplicitGenericParamConst.%T.loc6_34.2 (%T) [symbolic = %ptr (constants.%ptr)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @CallExplicitGenericParamConst.%ptr (%ptr) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc6_34.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_34.1 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:     %ExplicitGenericParam.specific_fn.loc7_3.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %ExplicitGenericParam.specific_fn.loc7_3.2 (constants.%ExplicitGenericParam.specific_fn)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamConst.%ptr (%ptr) = call %ExplicitGenericParam.specific_fn.loc7_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParamNonConst(%T.param_patt: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [concrete = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc4_39.2 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %ExplicitGenericParam.specific_fn.loc4_50.2 => constants.%ExplicitGenericParam.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(%T.loc4_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamConst(constants.%T) {
// CHECK:STDOUT:   %T.loc6_34.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_34.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(@CallExplicitGenericParamConst.%T.loc6_34.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_vs_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.type: type = fn_type @ExplicitAndAlsoDeduced [concrete]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced: %ExplicitAndAlsoDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn.41d: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T) [symbolic]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.type: type = fn_type @CallExplicitAndAlsoDeduced [concrete]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced: %CallExplicitAndAlsoDeduced.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn.720: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%A) [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.7ea: <witness> = complete_type_witness %ptr.6db [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .ExplicitAndAlsoDeduced = %ExplicitAndAlsoDeduced.decl
// CHECK:STDOUT:     .CallExplicitAndAlsoDeduced = %CallExplicitAndAlsoDeduced.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.decl: %ExplicitAndAlsoDeduced.type = fn_decl @ExplicitAndAlsoDeduced [concrete = constants.%ExplicitAndAlsoDeduced] {
// CHECK:STDOUT:     %T.patt.loc6_27.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_46: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc6_47.1: type = ptr_type %T [symbolic = %ptr.loc6_47.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc6_27.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc6_40: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.decl: %CallExplicitAndAlsoDeduced.type = fn_decl @CallExplicitAndAlsoDeduced [concrete = constants.%CallExplicitAndAlsoDeduced] {
// CHECK:STDOUT:     %return.patt: %ptr.6db = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.6db = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc10: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr: type = ptr_type %A [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitAndAlsoDeduced(%T.loc6_27.1: type) {
// CHECK:STDOUT:   %T.loc6_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_27.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_27.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc6_47.2: type = ptr_type @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) [symbolic = %ptr.loc6_47.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_43: <witness> = require_complete_type @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) [symbolic = %require_complete.loc6_43 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc6_38: <witness> = require_complete_type @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) [symbolic = %require_complete.loc6_38 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn.loc7_10.2: <specific function> = specific_function constants.%ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T.loc6_27.2) [symbolic = %ExplicitAndAlsoDeduced.specific_fn.loc7_10.2 (constants.%ExplicitAndAlsoDeduced.specific_fn.41d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc6_27.1: type, %x.param_patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T)) -> @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [concrete = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.specific_fn.loc7_10.1: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%T) [symbolic = %ExplicitAndAlsoDeduced.specific_fn.loc7_10.2 (constants.%ExplicitAndAlsoDeduced.specific_fn.41d)]
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.call: init @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = call %ExplicitAndAlsoDeduced.specific_fn.loc7_10.1(%x.ref)
// CHECK:STDOUT:     %.loc7_38.1: @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:     %.loc7_38.2: @ExplicitAndAlsoDeduced.%ptr.loc6_47.2 (%ptr.79f) = converted %ExplicitAndAlsoDeduced.call, %.loc7_38.1
// CHECK:STDOUT:     return %.loc7_38.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitAndAlsoDeduced() -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [concrete = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:   %A.ref.loc11: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc11_37.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%A) [concrete = constants.%ExplicitAndAlsoDeduced.specific_fn.720]
// CHECK:STDOUT:   %.loc11_37.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc11_37.3: init %A = class_init (), %.loc11_37.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc11_37.4: ref %A = temporary %.loc11_37.2, %.loc11_37.3
// CHECK:STDOUT:   %.loc11_37.5: ref %A = converted %.loc11_37.1, %.loc11_37.4
// CHECK:STDOUT:   %.loc11_37.6: %A = bind_value %.loc11_37.5
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.call: init %ptr.6db = call %ExplicitAndAlsoDeduced.specific_fn(%.loc11_37.6)
// CHECK:STDOUT:   %.loc11_39.1: %ptr.6db = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:   %.loc11_39.2: %ptr.6db = converted %ExplicitAndAlsoDeduced.call, %.loc11_39.1
// CHECK:STDOUT:   return %.loc11_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_47.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_43 => constants.%require_complete.6e5
// CHECK:STDOUT:   %require_complete.loc6_38 => constants.%require_complete.4ae
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn.loc7_10.2 => constants.%ExplicitAndAlsoDeduced.specific_fn.41d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(%T.loc6_27.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%A) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%A
// CHECK:STDOUT:   %ptr.loc6_47.2 => constants.%ptr.6db
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_43 => constants.%complete_type.7ea
// CHECK:STDOUT:   %require_complete.loc6_38 => constants.%complete_type.357
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.specific_fn.loc7_10.2 => constants.%ExplicitAndAlsoDeduced.specific_fn.720
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ImplicitGenericParam.type: type = fn_type @ImplicitGenericParam [concrete]
// CHECK:STDOUT:   %ImplicitGenericParam: %ImplicitGenericParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn.fc1: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %CallImplicitGenericParam.type: type = fn_type @CallImplicitGenericParam [concrete]
// CHECK:STDOUT:   %CallImplicitGenericParam: %CallImplicitGenericParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn.752: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.3d0: <witness> = complete_type_witness %ptr.235 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplicitGenericParam = %ImplicitGenericParam.decl
// CHECK:STDOUT:     .CallImplicitGenericParam = %CallImplicitGenericParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitGenericParam.decl: %ImplicitGenericParam.type = fn_decl @ImplicitGenericParam [concrete = constants.%ImplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitGenericParam.%T.loc4_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitGenericParam.%T.loc4_25.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_45.1: type = ptr_type %T [symbolic = %ptr.loc4_45.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ImplicitGenericParam.%T.loc4_25.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x: @ImplicitGenericParam.%T.loc4_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitGenericParam.decl: %CallImplicitGenericParam.type = fn_decl @CallImplicitGenericParam [concrete = constants.%CallImplicitGenericParam] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.235 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.235 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %i32.loc6_32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %ptr.235 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.235 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc4_45.2: type = ptr_type @ImplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %ptr.loc4_45.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_41: <witness> = require_complete_type @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) [symbolic = %require_complete.loc4_41 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc4_36: <witness> = require_complete_type @ImplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %require_complete.loc4_36 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn.loc4_56.2: <specific function> = specific_function constants.%ImplicitGenericParam, @ImplicitGenericParam(%T.loc4_25.2) [symbolic = %ImplicitGenericParam.specific_fn.loc4_56.2 (constants.%ImplicitGenericParam.specific_fn.fc1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_25.1: type](%x.param_patt: @ImplicitGenericParam.%T.loc4_25.2 (%T)) -> @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [concrete = constants.%ImplicitGenericParam]
// CHECK:STDOUT:     %x.ref: @ImplicitGenericParam.%T.loc4_25.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %ImplicitGenericParam.specific_fn.loc4_56.1: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(constants.%T) [symbolic = %ImplicitGenericParam.specific_fn.loc4_56.2 (constants.%ImplicitGenericParam.specific_fn.fc1)]
// CHECK:STDOUT:     %ImplicitGenericParam.call: init @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = call %ImplicitGenericParam.specific_fn.loc4_56.1(%x.ref)
// CHECK:STDOUT:     %.loc4_79.1: @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_79.2: @ImplicitGenericParam.%ptr.loc4_45.2 (%ptr.79f) = converted %ImplicitGenericParam.call, %.loc4_79.1
// CHECK:STDOUT:     return %.loc4_79.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitGenericParam(%n.param_patt: %i32) -> %ptr.235 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [concrete = constants.%ImplicitGenericParam]
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(constants.%i32) [concrete = constants.%ImplicitGenericParam.specific_fn.752]
// CHECK:STDOUT:   %ImplicitGenericParam.call: init %ptr.235 = call %ImplicitGenericParam.specific_fn(%n.ref)
// CHECK:STDOUT:   %.loc7_33.1: %ptr.235 = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_33.2: %ptr.235 = converted %ImplicitGenericParam.call, %.loc7_33.1
// CHECK:STDOUT:   return %.loc7_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc4_45.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_41 => constants.%require_complete.6e5
// CHECK:STDOUT:   %require_complete.loc4_36 => constants.%require_complete.4ae
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn.loc4_56.2 => constants.%ImplicitGenericParam.specific_fn.fc1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(%T.loc4_25.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %ptr.loc4_45.2 => constants.%ptr.235
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_41 => constants.%complete_type.3d0
// CHECK:STDOUT:   %require_complete.loc4_36 => constants.%complete_type.f8a
// CHECK:STDOUT:   %ImplicitGenericParam.specific_fn.loc4_56.2 => constants.%ImplicitGenericParam.specific_fn.752
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.f83: type = tuple_type (%T, %i32) [symbolic]
// CHECK:STDOUT:   %TupleParam.type: type = fn_type @TupleParam [concrete]
// CHECK:STDOUT:   %TupleParam: %TupleParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9c1: <witness> = require_complete_type %tuple.type.f83 [symbolic]
// CHECK:STDOUT:   %CallTupleParam.type: type = fn_type @CallTupleParam [concrete]
// CHECK:STDOUT:   %CallTupleParam: %CallTupleParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.type.4c8: type = tuple_type (Core.IntLiteral, %i32) [concrete]
// CHECK:STDOUT:   %TupleParam.specific_fn: <specific function> = specific_function %TupleParam, @TupleParam(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.4c8 = tuple_value (%int_1, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %complete_type.c49: <witness> = complete_type_witness %tuple.type.4c8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TupleParam = %TupleParam.decl
// CHECK:STDOUT:     .CallTupleParam = %CallTupleParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleParam.decl: %TupleParam.type = fn_decl @TupleParam [concrete = constants.%TupleParam] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TupleParam.%tuple.type (%tuple.type.f83) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TupleParam.%tuple.type (%tuple.type.f83) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TupleParam.%tuple.type (%tuple.type.f83) = value_param call_param0
// CHECK:STDOUT:     %.loc4_35.1: type = splice_block %.loc4_35.3 [symbolic = %tuple.type (constants.%tuple.type.f83)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_35.2: %tuple.type.24b = tuple_literal (%T.ref, %i32)
// CHECK:STDOUT:       %.loc4_35.3: type = converted %.loc4_35.2, constants.%tuple.type.f83 [symbolic = %tuple.type (constants.%tuple.type.f83)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TupleParam.%tuple.type (%tuple.type.f83) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTupleParam.decl: %CallTupleParam.type = fn_decl @CallTupleParam [concrete = constants.%CallTupleParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TupleParam(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@TupleParam.%T.loc4_15.2 (%T), %i32) [symbolic = %tuple.type (constants.%tuple.type.f83)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @TupleParam.%tuple.type (%tuple.type.f83) [symbolic = %require_complete (constants.%require_complete.9c1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @TupleParam.%tuple.type (%tuple.type.f83)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTupleParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TupleParam.ref: %TupleParam.type = name_ref TupleParam, file.%TupleParam.decl [concrete = constants.%TupleParam]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc7_19.1: %tuple.type.f94 = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:   %TupleParam.specific_fn: <specific function> = specific_function %TupleParam.ref, @TupleParam(Core.IntLiteral) [concrete = constants.%TupleParam.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc7_19.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_19.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc7_19.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc7_19.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc7_19.3: %i32 = converted %int_2, %.loc7_19.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple: %tuple.type.4c8 = tuple_value (%int_1, %.loc7_19.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc7_19.4: %tuple.type.4c8 = converted %.loc7_19.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %TupleParam.call: init %empty_tuple.type = call %TupleParam.specific_fn(%.loc7_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.f83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc4_15.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc4_15.2 => Core.IntLiteral
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.4c8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.46e: type = struct_type {.a: %T, .b: %i32} [symbolic]
// CHECK:STDOUT:   %StructParam.type: type = fn_type @StructParam [concrete]
// CHECK:STDOUT:   %StructParam: %StructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.23a: <witness> = require_complete_type %struct_type.a.b.46e [symbolic]
// CHECK:STDOUT:   %CallStructParam.type: type = fn_type @CallStructParam [concrete]
// CHECK:STDOUT:   %CallStructParam: %CallStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct_type.a.b.a13: type = struct_type {.a: Core.IntLiteral, .b: %i32} [concrete]
// CHECK:STDOUT:   %StructParam.specific_fn: <specific function> = specific_function %StructParam, @StructParam(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.a13 = struct_value (%int_1, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %complete_type.a4a: <witness> = complete_type_witness %struct_type.a.b.a13 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .StructParam = %StructParam.decl
// CHECK:STDOUT:     .CallStructParam = %CallStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %StructParam.decl: %StructParam.type = fn_decl @StructParam [concrete = constants.%StructParam] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %struct_type.a.b.loc4_44.1 [symbolic = %struct_type.a.b.loc4_44.2 (constants.%struct_type.a.b.46e)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b.loc4_44.1: type = struct_type {.a: %T, .b: %i32} [symbolic = %struct_type.a.b.loc4_44.2 (constants.%struct_type.a.b.46e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallStructParam.decl: %CallStructParam.type = fn_decl @CallStructParam [concrete = constants.%CallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StructParam(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.a.b.loc4_44.2: type = struct_type {.a: @StructParam.%T.loc4_16.2 (%T), .b: %i32} [symbolic = %struct_type.a.b.loc4_44.2 (constants.%struct_type.a.b.46e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e) [symbolic = %require_complete (constants.%require_complete.23a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_16.1: type](%x.param_patt: @StructParam.%struct_type.a.b.loc4_44.2 (%struct_type.a.b.46e)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %StructParam.ref: %StructParam.type = name_ref StructParam, file.%StructParam.decl [concrete = constants.%StructParam]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc7_30.1: %struct_type.a.b.cfd = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %StructParam.specific_fn: <specific function> = specific_function %StructParam.ref, @StructParam(Core.IntLiteral) [concrete = constants.%StructParam.specific_fn]
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc7_30.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_30.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc7_30.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc7_30.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc7_30.3: %i32 = converted %int_2, %.loc7_30.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %struct: %struct_type.a.b.a13 = struct_value (%int_1, %.loc7_30.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_30.4: %struct_type.a.b.a13 = converted %.loc7_30.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %StructParam.call: init %empty_tuple.type = call %StructParam.specific_fn(%.loc7_30.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %struct_type.a.b.loc4_44.2 => constants.%struct_type.a.b.46e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %struct_type.a.b.loc4_44.2 => constants.%struct_type.a.b.a13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.a4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_bigger_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.c.d.e: type = struct_type {.c: %T, .d: %i32, .e: %i32} [symbolic]
// CHECK:STDOUT:   %BigStructParam.type: type = fn_type @BigStructParam [concrete]
// CHECK:STDOUT:   %BigStructParam: %BigStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.694: <witness> = require_complete_type %struct_type.c.d.e [symbolic]
// CHECK:STDOUT:   %CallBigStructParam.type: type = fn_type @CallBigStructParam [concrete]
// CHECK:STDOUT:   %CallBigStructParam: %CallBigStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.c.d: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .BigStructParam = %BigStructParam.decl
// CHECK:STDOUT:     .CallBigStructParam = %CallBigStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %BigStructParam.decl: %BigStructParam.type = fn_decl @BigStructParam [concrete = constants.%BigStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %struct_type.c.d.e.loc4_56.1 [symbolic = %struct_type.c.d.e.loc4_56.2 (constants.%struct_type.c.d.e)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_19.1 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:       %int_32.loc4_44: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_44: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc4_53: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_53: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.c.d.e.loc4_56.1: type = struct_type {.c: %T, .d: %i32, .e: %i32} [symbolic = %struct_type.c.d.e.loc4_56.2 (constants.%struct_type.c.d.e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallBigStructParam.decl: %CallBigStructParam.type = fn_decl @CallBigStructParam [concrete = constants.%CallBigStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BigStructParam(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.c.d.e.loc4_56.2: type = struct_type {.c: @BigStructParam.%T.loc4_19.2 (%T), .d: %i32, .e: %i32} [symbolic = %struct_type.c.d.e.loc4_56.2 (constants.%struct_type.c.d.e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e) [symbolic = %require_complete (constants.%require_complete.694)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_19.1: type](%x.param_patt: @BigStructParam.%struct_type.c.d.e.loc4_56.2 (%struct_type.c.d.e)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBigStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BigStructParam.ref: %BigStructParam.type = name_ref BigStructParam, file.%BigStructParam.decl [concrete = constants.%BigStructParam]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc14: %struct_type.c.d = struct_literal (%int_3, %int_4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BigStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %struct_type.c.d.e.loc4_56.2 => constants.%struct_type.c.d.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_smaller_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.f.g: type = struct_type {.f: %T, .g: %i32} [symbolic]
// CHECK:STDOUT:   %SmallStructParam.type: type = fn_type @SmallStructParam [concrete]
// CHECK:STDOUT:   %SmallStructParam: %SmallStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.dad: <witness> = require_complete_type %struct_type.f.g [symbolic]
// CHECK:STDOUT:   %CallSmallStructParam.type: type = fn_type @CallSmallStructParam [concrete]
// CHECK:STDOUT:   %CallSmallStructParam: %CallSmallStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %struct_type.f.g.h: type = struct_type {.f: Core.IntLiteral, .g: Core.IntLiteral, .h: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SmallStructParam = %SmallStructParam.decl
// CHECK:STDOUT:     .CallSmallStructParam = %CallSmallStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SmallStructParam.decl: %SmallStructParam.type = fn_decl @SmallStructParam [concrete = constants.%SmallStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %struct_type.f.g.loc4_49.1 [symbolic = %struct_type.f.g.loc4_49.2 (constants.%struct_type.f.g)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_21.1 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.f.g.loc4_49.1: type = struct_type {.f: %T, .g: %i32} [symbolic = %struct_type.f.g.loc4_49.2 (constants.%struct_type.f.g)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallSmallStructParam.decl: %CallSmallStructParam.type = fn_decl @CallSmallStructParam [concrete = constants.%CallSmallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SmallStructParam(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.f.g.loc4_49.2: type = struct_type {.f: @SmallStructParam.%T.loc4_21.2 (%T), .g: %i32} [symbolic = %struct_type.f.g.loc4_49.2 (constants.%struct_type.f.g)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g) [symbolic = %require_complete (constants.%require_complete.dad)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_21.1: type](%x.param_patt: @SmallStructParam.%struct_type.f.g.loc4_49.2 (%struct_type.f.g)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSmallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SmallStructParam.ref: %SmallStructParam.type = name_ref SmallStructParam, file.%SmallStructParam.decl [concrete = constants.%SmallStructParam]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc14: %struct_type.f.g.h = struct_literal (%int_5, %int_6, %int_7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SmallStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %struct_type.f.g.loc4_49.2 => constants.%struct_type.f.g
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_struct_wrong_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.i.different: type = struct_type {.i: %T, .different: %i32} [symbolic]
// CHECK:STDOUT:   %WrongNameStructParam.type: type = fn_type @WrongNameStructParam [concrete]
// CHECK:STDOUT:   %WrongNameStructParam: %WrongNameStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.411: <witness> = require_complete_type %struct_type.i.different [symbolic]
// CHECK:STDOUT:   %CallWrongNameStructParam.type: type = fn_type @CallWrongNameStructParam [concrete]
// CHECK:STDOUT:   %CallWrongNameStructParam: %CallWrongNameStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %struct_type.i.j: type = struct_type {.i: Core.IntLiteral, .j: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrongNameStructParam = %WrongNameStructParam.decl
// CHECK:STDOUT:     .CallWrongNameStructParam = %CallWrongNameStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongNameStructParam.decl: %WrongNameStructParam.type = fn_decl @WrongNameStructParam [concrete = constants.%WrongNameStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %struct_type.i.different.loc4_61.1 [symbolic = %struct_type.i.different.loc4_61.2 (constants.%struct_type.i.different)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.i.different.loc4_61.1: type = struct_type {.i: %T, .different: %i32} [symbolic = %struct_type.i.different.loc4_61.2 (constants.%struct_type.i.different)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongNameStructParam.decl: %CallWrongNameStructParam.type = fn_decl @CallWrongNameStructParam [concrete = constants.%CallWrongNameStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongNameStructParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.i.different.loc4_61.2: type = struct_type {.i: @WrongNameStructParam.%T.loc4_25.2 (%T), .different: %i32} [symbolic = %struct_type.i.different.loc4_61.2 (constants.%struct_type.i.different)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different) [symbolic = %require_complete (constants.%require_complete.411)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_25.1: type](%x.param_patt: @WrongNameStructParam.%struct_type.i.different.loc4_61.2 (%struct_type.i.different)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongNameStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongNameStructParam.ref: %WrongNameStructParam.type = name_ref WrongNameStructParam, file.%WrongNameStructParam.decl [concrete = constants.%WrongNameStructParam]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc14: %struct_type.i.j = struct_literal (%int_8, %int_9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongNameStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %struct_type.i.different.loc4_61.2 => constants.%struct_type.i.different
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_deduce_struct_wrong_order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.first.second: type = struct_type {.first: %T, .second: %i32} [symbolic]
// CHECK:STDOUT:   %WrongOrderStructParam.type: type = fn_type @WrongOrderStructParam [concrete]
// CHECK:STDOUT:   %WrongOrderStructParam: %WrongOrderStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d6e: <witness> = require_complete_type %struct_type.first.second [symbolic]
// CHECK:STDOUT:   %CallWrongOrderStructParam.type: type = fn_type @CallWrongOrderStructParam [concrete]
// CHECK:STDOUT:   %CallWrongOrderStructParam: %CallWrongOrderStructParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %struct_type.second.first: type = struct_type {.second: Core.IntLiteral, .first: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrongOrderStructParam = %WrongOrderStructParam.decl
// CHECK:STDOUT:     .CallWrongOrderStructParam = %CallWrongOrderStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongOrderStructParam.decl: %WrongOrderStructParam.type = fn_decl @WrongOrderStructParam [concrete = constants.%WrongOrderStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_26.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %struct_type.first.second.loc4_63.1 [symbolic = %struct_type.first.second.loc4_63.2 (constants.%struct_type.first.second)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc4_26.1 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.first.second.loc4_63.1: type = struct_type {.first: %T, .second: %i32} [symbolic = %struct_type.first.second.loc4_63.2 (constants.%struct_type.first.second)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongOrderStructParam.decl: %CallWrongOrderStructParam.type = fn_decl @CallWrongOrderStructParam [concrete = constants.%CallWrongOrderStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongOrderStructParam(%T.loc4_26.1: type) {
// CHECK:STDOUT:   %T.loc4_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   %struct_type.first.second.loc4_63.2: type = struct_type {.first: @WrongOrderStructParam.%T.loc4_26.2 (%T), .second: %i32} [symbolic = %struct_type.first.second.loc4_63.2 (constants.%struct_type.first.second)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second) [symbolic = %require_complete (constants.%require_complete.d6e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_26.1: type](%x.param_patt: @WrongOrderStructParam.%struct_type.first.second.loc4_63.2 (%struct_type.first.second)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongOrderStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongOrderStructParam.ref: %WrongOrderStructParam.type = name_ref WrongOrderStructParam, file.%WrongOrderStructParam.decl [concrete = constants.%WrongOrderStructParam]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc14: %struct_type.second.first = struct_literal (%int_11, %int_10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongOrderStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %struct_type.first.second.loc4_63.2 => constants.%struct_type.first.second
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [concrete]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [concrete]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplicitNotDeducible = %ImplicitNotDeducible.decl
// CHECK:STDOUT:     .CallImplicitNotDeducible = %CallImplicitNotDeducible.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitNotDeducible.decl: %ImplicitNotDeducible.type = fn_decl @ImplicitNotDeducible [concrete = constants.%ImplicitNotDeducible] {
// CHECK:STDOUT:     %T.patt.loc6_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc6_35.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_35.2 (constants.%U.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @ImplicitNotDeducible.%U.loc6_35.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitNotDeducible.%U.loc6_35.2 (%U) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc6_35.1 [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:     %T.loc6_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc6_35.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:     %x.param: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_25.1 [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ImplicitNotDeducible.%U.loc6_35.2 (%U) = out_param call_param1
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%U.loc6_35.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [concrete = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc6_25.1: type, %U.loc6_35.1: type) {
// CHECK:STDOUT:   %T.loc6_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc6_35.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_35.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_35.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_25.1: type, %U.patt.loc6_35.1: type](%x.param_patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T)) -> @ImplicitNotDeducible.%U.loc6_35.2 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [concrete = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc6_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_25.2 => constants.%T
// CHECK:STDOUT:   %U.loc6_35.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc6_35.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [concrete]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [concrete]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplicitNotDeducible = %ImplicitNotDeducible.decl
// CHECK:STDOUT:     .CallImplicitNotDeducible = %CallImplicitNotDeducible.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitNotDeducible.decl: %ImplicitNotDeducible.type = fn_decl @ImplicitNotDeducible [concrete = constants.%ImplicitNotDeducible] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %y.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param_pattern %y.patt, call_param1
// CHECK:STDOUT:     %return.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_50: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %y: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref @ImplicitNotDeducible.%T.loc4_25.2 (%T) = out_param call_param2
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%T.loc4_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [concrete = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_25.1: type](%x.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T), %y.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T)) -> @ImplicitNotDeducible.%T.loc4_25.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [concrete = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc14: %struct_type.x = struct_literal (%int_12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
