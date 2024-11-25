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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE+6]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   ImplicitNotDeducible(42, {.x = 12});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn ImplicitNotDeducible[T:! type](x: T, y: T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ImplicitNotDeducible(42, {.x = 12});
}

// CHECK:STDOUT: --- deduce_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %CallExplicitGenericParam.type: type = fn_type @CallExplicitGenericParam [template]
// CHECK:STDOUT:   %CallExplicitGenericParam: %CallExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%i32) [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.type: type = fn_type @CallExplicitGenericParamWithGenericArg [template]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg: %CallExplicitGenericParamWithGenericArg.type = struct_value () [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%.6) [symbolic]
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
// CHECK:STDOUT:     .ExplicitGenericParam = %ExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParam = %CallExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParamWithGenericArg = %CallExplicitGenericParamWithGenericArg.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ExplicitGenericParam.decl: %ExplicitGenericParam.type = fn_decl @ExplicitGenericParam [template = constants.%ExplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @ExplicitGenericParam.%.loc4_39.2 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitGenericParam.%.loc4_39.2 (%.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_39.1: type = ptr_type %T [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @ExplicitGenericParam.%.loc4_39.2 (%.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @ExplicitGenericParam.%.loc4_39.2 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParam.decl: %CallExplicitGenericParam.type = fn_decl @CallExplicitGenericParam [template = constants.%CallExplicitGenericParam] {
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_34: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_37.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_37.2: type = converted %int.make_type_signed.loc6, %.loc6_37.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_37.3: type = ptr_type %i32 [template = constants.%.4]
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.decl: %CallExplicitGenericParamWithGenericArg.type = fn_decl @CallExplicitGenericParamWithGenericArg [template = constants.%CallExplicitGenericParamWithGenericArg] {
// CHECK:STDOUT:     %T.patt.loc10_43.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_43.1, runtime_param<invalid> [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10_62.1: type = struct_type {.a: %T} [symbolic = %.loc10_62.2 (constants.%.6)]
// CHECK:STDOUT:     %.loc10_63.1: type = ptr_type %.6 [symbolic = %.loc10_63.2 (constants.%.7)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_43.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_50.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @ExplicitGenericParam.%.loc4_39.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc4_50.2 (constants.%.2)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%.loc4_39.2 (%.1) = call %.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%.loc4_39.2 (%.1) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%.loc4_39.2 (%.1) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParam() -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %.loc7_31: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_31) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_signed.loc7, %.loc7_30.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%i32) [template = constants.%.5]
// CHECK:STDOUT:   %ExplicitGenericParam.call: init %.4 = call %.loc7_10()
// CHECK:STDOUT:   %.loc7_35.1: %.4 = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_35.2: %.4 = converted %ExplicitGenericParam.call, %.loc7_35.1
// CHECK:STDOUT:   return %.loc7_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamWithGenericArg(%T.loc10_43.1: type) {
// CHECK:STDOUT:   %T.loc10_43.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_43.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc10_62.2: type = struct_type {.a: @CallExplicitGenericParamWithGenericArg.%T.loc10_43.2 (%T)} [symbolic = %.loc10_62.2 (constants.%.6)]
// CHECK:STDOUT:   %.loc10_63.2: type = ptr_type @CallExplicitGenericParamWithGenericArg.%.loc10_62.2 (%.6) [symbolic = %.loc10_63.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_10.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%.loc10_62.2) [symbolic = %.loc11_10.2 (constants.%.8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc11: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %.loc11_37: type = struct_type {.a: %T} [symbolic = %.loc10_62.2 (constants.%.6)]
// CHECK:STDOUT:     %.loc11_10.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%.6) [symbolic = %.loc11_10.2 (constants.%.8)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = call %.loc11_10.1()
// CHECK:STDOUT:     %.loc11_39.1: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc11_39.2: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.7) = converted %ExplicitGenericParam.call, %.loc11_39.1
// CHECK:STDOUT:     return %.loc11_39.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(%T.loc4_25.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamWithGenericArg(constants.%T) {
// CHECK:STDOUT:   %T.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %.loc10_62.2 => constants.%.6
// CHECK:STDOUT:   %.loc10_63.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%.6) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%.6
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%.6
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(@CallExplicitGenericParamWithGenericArg.%.loc10_62.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%.6
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%.6
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_explicit_non_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamConst.type: type = fn_type @CallExplicitGenericParamConst [template]
// CHECK:STDOUT:   %CallExplicitGenericParamConst: %CallExplicitGenericParamConst.type = struct_value () [template]
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst.type: type = fn_type @CallExplicitGenericParamNonConst [template]
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst: %CallExplicitGenericParamNonConst.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ExplicitGenericParam = %ExplicitGenericParam.decl
// CHECK:STDOUT:     .CallExplicitGenericParamConst = %CallExplicitGenericParamConst.decl
// CHECK:STDOUT:     .CallExplicitGenericParamNonConst = %CallExplicitGenericParamNonConst.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ExplicitGenericParam.decl: %ExplicitGenericParam.type = fn_decl @ExplicitGenericParam [template = constants.%ExplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @ExplicitGenericParam.%.loc4_39.2 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitGenericParam.%.loc4_39.2 (%.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_39.1: type = ptr_type %T [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @ExplicitGenericParam.%.loc4_39.2 (%.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @ExplicitGenericParam.%.loc4_39.2 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamConst.decl: %CallExplicitGenericParamConst.type = fn_decl @CallExplicitGenericParamConst [template = constants.%CallExplicitGenericParamConst] {
// CHECK:STDOUT:     %T.patt.loc6_34.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_34.1, runtime_param<invalid> [symbolic = %T.patt.loc6_34.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_34.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamNonConst.decl: %CallExplicitGenericParamNonConst.type = fn_decl @CallExplicitGenericParamNonConst [template = constants.%CallExplicitGenericParamNonConst] {
// CHECK:STDOUT:     %T.patt: type = binding_pattern T
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param0
// CHECK:STDOUT:     %T: type = bind_name T, %T.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_50.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @ExplicitGenericParam.%.loc4_39.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc4_50.2 (constants.%.2)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%.loc4_39.2 (%.1) = call %.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%.loc4_39.2 (%.1) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%.loc4_39.2 (%.1) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamConst(%T.loc6_34.1: type) {
// CHECK:STDOUT:   %T.loc6_34.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_34.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_3.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc6_34.2) [symbolic = %.loc7_3.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc7_23: type = ptr_type @CallExplicitGenericParamConst.%T.loc6_34.2 (%T) [symbolic = %.loc7_23 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_34.1 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:     %.loc7_3.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc7_3.2 (constants.%.2)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamConst.%.loc7_23 (%.1) = call %.loc7_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParamNonConst(%T.param_patt: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(%T.loc4_25.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamConst(constants.%T) {
// CHECK:STDOUT:   %T.loc6_34.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_34.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(@CallExplicitGenericParamConst.%T.loc6_34.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_vs_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.type: type = fn_type @ExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced: %ExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %A [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.type: type = fn_type @CallExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced: %CallExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.6: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%A) [template]
// CHECK:STDOUT:   %struct: %A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .ExplicitAndAlsoDeduced = %ExplicitAndAlsoDeduced.decl
// CHECK:STDOUT:     .CallExplicitAndAlsoDeduced = %CallExplicitAndAlsoDeduced.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.decl: %ExplicitAndAlsoDeduced.type = fn_decl @ExplicitAndAlsoDeduced [template = constants.%ExplicitAndAlsoDeduced] {
// CHECK:STDOUT:     %T.patt.loc6_27.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_27.1, runtime_param<invalid> [symbolic = %T.patt.loc6_27.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_40: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc6_46: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_47.1: type = ptr_type %T [symbolic = %.loc6_47.2 (constants.%.3)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_27.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.decl: %CallExplicitAndAlsoDeduced.type = fn_decl @CallExplicitAndAlsoDeduced [template = constants.%CallExplicitAndAlsoDeduced] {
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc10: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc10_37: type = ptr_type %A [template = constants.%.5]
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitAndAlsoDeduced(%T.loc6_27.1: type) {
// CHECK:STDOUT:   %T.loc6_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_27.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_27.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_47.2: type = ptr_type @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) [symbolic = %.loc6_47.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_10.2: <specific function> = specific_function constants.%ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T.loc6_27.2) [symbolic = %.loc7_10.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T)) -> @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [template = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_10.1: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%T) [symbolic = %.loc7_10.2 (constants.%.4)]
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.call: init @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = call %.loc7_10.1(%x.ref)
// CHECK:STDOUT:     %.loc7_38.1: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:     %.loc7_38.2: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = converted %ExplicitAndAlsoDeduced.call, %.loc7_38.1
// CHECK:STDOUT:     return %.loc7_38.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitAndAlsoDeduced() -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [template = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:   %A.ref.loc11: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc11_37.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc11_10: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%A) [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_37.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc11_37.3: init %A = class_init (), %.loc11_37.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_37.4: ref %A = temporary %.loc11_37.2, %.loc11_37.3
// CHECK:STDOUT:   %.loc11_37.5: ref %A = converted %.loc11_37.1, %.loc11_37.4
// CHECK:STDOUT:   %.loc11_37.6: %A = bind_value %.loc11_37.5
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.call: init %.5 = call %.loc11_10(%.loc11_37.6)
// CHECK:STDOUT:   %.loc11_39.1: %.5 = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:   %.loc11_39.2: %.5 = converted %ExplicitAndAlsoDeduced.call, %.loc11_39.1
// CHECK:STDOUT:   return %.loc11_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %.loc6_47.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_10.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(%T.loc6_27.2) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %.loc6_47.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%A) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%A
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%A
// CHECK:STDOUT:   %.loc6_47.2 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_10.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ImplicitGenericParam.type: type = fn_type @ImplicitGenericParam [template]
// CHECK:STDOUT:   %ImplicitGenericParam: %ImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %CallImplicitGenericParam.type: type = fn_type @CallImplicitGenericParam [template]
// CHECK:STDOUT:   %CallImplicitGenericParam: %CallImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%i32) [template]
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
// CHECK:STDOUT:     .ImplicitGenericParam = %ImplicitGenericParam.decl
// CHECK:STDOUT:     .CallImplicitGenericParam = %CallImplicitGenericParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitGenericParam.decl: %ImplicitGenericParam.type = fn_decl @ImplicitGenericParam [template = constants.%ImplicitGenericParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitGenericParam.%T.loc4_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitGenericParam.%T.loc4_25.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @ImplicitGenericParam.%.loc4_45.2 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitGenericParam.%.loc4_45.2 (%.1) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_45.1: type = ptr_type %T [symbolic = %.loc4_45.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ImplicitGenericParam.%T.loc4_25.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitGenericParam.%T.loc4_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ImplicitGenericParam.%.loc4_45.2 (%.1) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @ImplicitGenericParam.%.loc4_45.2 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitGenericParam.decl: %CallImplicitGenericParam.type = fn_decl @CallImplicitGenericParam [template = constants.%CallImplicitGenericParam] {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_32.1: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc6_32: init type = call constants.%Int(%.loc6_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_32.2: type = value_of_initializer %int.make_type_signed.loc6_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_32.3: type = converted %int.make_type_signed.loc6_32, %.loc6_32.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_40: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed.loc6_40: init type = call constants.%Int(%.loc6_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_signed.loc6_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_signed.loc6_40, %.loc6_43.1 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_43.3: type = ptr_type %i32 [template = constants.%.4]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitGenericParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_45.2: type = ptr_type @ImplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %.loc4_45.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_56.2: <specific function> = specific_function constants.%ImplicitGenericParam, @ImplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_56.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @ImplicitGenericParam.%T.loc4_25.2 (%T)) -> @ImplicitGenericParam.%.loc4_45.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:     %x.ref: @ImplicitGenericParam.%T.loc4_25.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_56.1: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(constants.%T) [symbolic = %.loc4_56.2 (constants.%.2)]
// CHECK:STDOUT:     %ImplicitGenericParam.call: init @ImplicitGenericParam.%.loc4_45.2 (%.1) = call %.loc4_56.1(%x.ref)
// CHECK:STDOUT:     %.loc4_79.1: @ImplicitGenericParam.%.loc4_45.2 (%.1) = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_79.2: @ImplicitGenericParam.%.loc4_45.2 (%.1) = converted %ImplicitGenericParam.call, %.loc4_79.1
// CHECK:STDOUT:     return %.loc4_79.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitGenericParam(%n.param_patt: %i32) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(constants.%i32) [template = constants.%.5]
// CHECK:STDOUT:   %ImplicitGenericParam.call: init %.4 = call %.loc7_10(%n.ref)
// CHECK:STDOUT:   %.loc7_33.1: %.4 = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_33.2: %.4 = converted %ImplicitGenericParam.call, %.loc7_33.1
// CHECK:STDOUT:   return %.loc7_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_45.2 => constants.%.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(%T.loc4_25.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_45.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%i32
// CHECK:STDOUT:   %.loc4_45.2 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%T, %i32) [symbolic]
// CHECK:STDOUT:   %TupleParam.type: type = fn_type @TupleParam [template]
// CHECK:STDOUT:   %TupleParam: %TupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallTupleParam.type: type = fn_type @CallTupleParam [template]
// CHECK:STDOUT:   %CallTupleParam: %CallTupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (Core.IntLiteral, %i32) [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %TupleParam, @TupleParam(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.4 = tuple_value (%.3, %.33) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TupleParam = %TupleParam.decl
// CHECK:STDOUT:     .CallTupleParam = %CallTupleParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleParam.decl: %TupleParam.type = fn_decl @TupleParam [template = constants.%TupleParam] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_15.1, runtime_param<invalid> [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TupleParam.%tuple.type (%tuple.type.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TupleParam.%tuple.type (%tuple.type.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_35.1: %tuple.type.1 = tuple_literal (%T.ref, %int.make_type_signed)
// CHECK:STDOUT:     %.loc4_35.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_35.3: type = converted %int.make_type_signed, %.loc4_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_35.4: type = converted %.loc4_35.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TupleParam.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @TupleParam.%tuple.type (%tuple.type.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTupleParam.decl: %CallTupleParam.type = fn_decl @CallTupleParam [template = constants.%CallTupleParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TupleParam(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@TupleParam.%T.loc4_15.2 (%T), %i32) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @TupleParam.%tuple.type (%tuple.type.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTupleParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TupleParam.ref: %TupleParam.type = name_ref TupleParam, file.%TupleParam.decl [template = constants.%TupleParam]
// CHECK:STDOUT:   %.loc7_15: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_18: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_19.1: %tuple.type.3 = tuple_literal (%.loc7_15, %.loc7_18)
// CHECK:STDOUT:   %.loc7_3: <specific function> = specific_function %TupleParam.ref, @TupleParam(Core.IntLiteral) [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_19.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_19.3: <bound method> = bound_method %.loc7_18, %.loc7_19.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc7_19.4: <specific function> = specific_function %.loc7_19.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc7_19.4(%.loc7_18) [template = constants.%.33]
// CHECK:STDOUT:   %.loc7_19.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   %.loc7_19.6: %i32 = converted %.loc7_18, %.loc7_19.5 [template = constants.%.33]
// CHECK:STDOUT:   %tuple: %tuple.type.4 = tuple_value (%.loc7_15, %.loc7_19.6) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_19.7: %tuple.type.4 = converted %.loc7_19.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %TupleParam.call: init %empty_tuple.type = call %.loc7_3(%.loc7_19.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc4_15.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc4_15.2 => Core.IntLiteral
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %T, .b: %i32} [symbolic]
// CHECK:STDOUT:   %StructParam.type: type = fn_type @StructParam [template]
// CHECK:STDOUT:   %StructParam: %StructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallStructParam.type: type = fn_type @CallStructParam [template]
// CHECK:STDOUT:   %CallStructParam: %CallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: Core.IntLiteral, .b: %i32} [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %StructParam, @StructParam(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.7 = struct_value (%.4, %.36) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .StructParam = %StructParam.decl
// CHECK:STDOUT:     .CallStructParam = %CallStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %StructParam.decl: %StructParam.type = fn_decl @StructParam [template = constants.%StructParam] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_16.1, runtime_param<invalid> [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @StructParam.%.loc4_44.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @StructParam.%.loc4_44.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_41.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_41.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_41.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_41.3: type = converted %int.make_type_signed, %.loc4_41.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_44.1: type = struct_type {.a: %T, .b: %i32} [symbolic = %.loc4_44.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @StructParam.%.loc4_44.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @StructParam.%.loc4_44.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallStructParam.decl: %CallStructParam.type = fn_decl @CallStructParam [template = constants.%CallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StructParam(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_44.2: type = struct_type {.a: @StructParam.%T.loc4_16.2 (%T), .b: %i32} [symbolic = %.loc4_44.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @StructParam.%.loc4_44.2 (%.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %StructParam.ref: %StructParam.type = name_ref StructParam, file.%StructParam.decl [template = constants.%StructParam]
// CHECK:STDOUT:   %.loc7_21: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_29: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_30.1: %.6 = struct_literal (%.loc7_21, %.loc7_29)
// CHECK:STDOUT:   %.loc7_3: <specific function> = specific_function %StructParam.ref, @StructParam(Core.IntLiteral) [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_30.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_30.3: <bound method> = bound_method %.loc7_29, %.loc7_30.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_30.4: <specific function> = specific_function %.loc7_30.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc7_30.4(%.loc7_29) [template = constants.%.36]
// CHECK:STDOUT:   %.loc7_30.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.36]
// CHECK:STDOUT:   %.loc7_30.6: %i32 = converted %.loc7_29, %.loc7_30.5 [template = constants.%.36]
// CHECK:STDOUT:   %struct: %.7 = struct_value (%.loc7_21, %.loc7_30.6) [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_30.7: %.7 = converted %.loc7_30.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %StructParam.call: init %empty_tuple.type = call %.loc7_3(%.loc7_30.7)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %.loc4_44.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %.loc4_44.2 => constants.%.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_bigger_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.c: %T, .d: %i32, .e: %i32} [symbolic]
// CHECK:STDOUT:   %BigStructParam.type: type = fn_type @BigStructParam [template]
// CHECK:STDOUT:   %BigStructParam: %BigStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallBigStructParam.type: type = fn_type @CallBigStructParam [template]
// CHECK:STDOUT:   %CallBigStructParam: %CallBigStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [template]
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
// CHECK:STDOUT:     .BigStructParam = %BigStructParam.decl
// CHECK:STDOUT:     .CallBigStructParam = %CallBigStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %BigStructParam.decl: %BigStructParam.type = fn_decl @BigStructParam [template = constants.%BigStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_19.1, runtime_param<invalid> [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @BigStructParam.%.loc4_56.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @BigStructParam.%.loc4_56.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_19.1 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_44.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_44: init type = call constants.%Int(%.loc4_44.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_44.2: type = value_of_initializer %int.make_type_signed.loc4_44 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_44.3: type = converted %int.make_type_signed.loc4_44, %.loc4_44.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_53.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_53: init type = call constants.%Int(%.loc4_53.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_53.2: type = value_of_initializer %int.make_type_signed.loc4_53 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_53.3: type = converted %int.make_type_signed.loc4_53, %.loc4_53.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_56.1: type = struct_type {.c: %T, .d: %i32, .e: %i32} [symbolic = %.loc4_56.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @BigStructParam.%.loc4_56.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @BigStructParam.%.loc4_56.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallBigStructParam.decl: %CallBigStructParam.type = fn_decl @CallBigStructParam [template = constants.%CallBigStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BigStructParam(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_56.2: type = struct_type {.c: @BigStructParam.%T.loc4_19.2 (%T), .d: %i32, .e: %i32} [symbolic = %.loc4_56.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @BigStructParam.%.loc4_56.2 (%.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBigStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BigStructParam.ref: %BigStructParam.type = name_ref BigStructParam, file.%BigStructParam.decl [template = constants.%BigStructParam]
// CHECK:STDOUT:   %.loc14_24: Core.IntLiteral = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_32: Core.IntLiteral = int_value 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_33: %.6 = struct_literal (%.loc14_24, %.loc14_32)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BigStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_smaller_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.f: %T, .g: %i32} [symbolic]
// CHECK:STDOUT:   %SmallStructParam.type: type = fn_type @SmallStructParam [template]
// CHECK:STDOUT:   %SmallStructParam: %SmallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallSmallStructParam.type: type = fn_type @CallSmallStructParam [template]
// CHECK:STDOUT:   %CallSmallStructParam: %CallSmallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.f: Core.IntLiteral, .g: Core.IntLiteral, .h: Core.IntLiteral} [template]
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
// CHECK:STDOUT:     .SmallStructParam = %SmallStructParam.decl
// CHECK:STDOUT:     .CallSmallStructParam = %CallSmallStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SmallStructParam.decl: %SmallStructParam.type = fn_decl @SmallStructParam [template = constants.%SmallStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_21.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_21.1, runtime_param<invalid> [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @SmallStructParam.%.loc4_49.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @SmallStructParam.%.loc4_49.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_21.1 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_46.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_46.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_46.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_46.3: type = converted %int.make_type_signed, %.loc4_46.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_49.1: type = struct_type {.f: %T, .g: %i32} [symbolic = %.loc4_49.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @SmallStructParam.%.loc4_49.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @SmallStructParam.%.loc4_49.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallSmallStructParam.decl: %CallSmallStructParam.type = fn_decl @CallSmallStructParam [template = constants.%CallSmallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SmallStructParam(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_49.2: type = struct_type {.f: @SmallStructParam.%T.loc4_21.2 (%T), .g: %i32} [symbolic = %.loc4_49.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @SmallStructParam.%.loc4_49.2 (%.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSmallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SmallStructParam.ref: %SmallStructParam.type = name_ref SmallStructParam, file.%SmallStructParam.decl [template = constants.%SmallStructParam]
// CHECK:STDOUT:   %.loc14_26: Core.IntLiteral = int_value 5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_34: Core.IntLiteral = int_value 6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_42: Core.IntLiteral = int_value 7 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_43: %.7 = struct_literal (%.loc14_26, %.loc14_34, %.loc14_42)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SmallStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %.loc4_49.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_struct_wrong_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.i: %T, .different: %i32} [symbolic]
// CHECK:STDOUT:   %WrongNameStructParam.type: type = fn_type @WrongNameStructParam [template]
// CHECK:STDOUT:   %WrongNameStructParam: %WrongNameStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallWrongNameStructParam.type: type = fn_type @CallWrongNameStructParam [template]
// CHECK:STDOUT:   %CallWrongNameStructParam: %CallWrongNameStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.i: Core.IntLiteral, .j: Core.IntLiteral} [template]
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
// CHECK:STDOUT:     .WrongNameStructParam = %WrongNameStructParam.decl
// CHECK:STDOUT:     .CallWrongNameStructParam = %CallWrongNameStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongNameStructParam.decl: %WrongNameStructParam.type = fn_decl @WrongNameStructParam [template = constants.%WrongNameStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @WrongNameStructParam.%.loc4_61.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongNameStructParam.%.loc4_61.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_58.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_58.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_58.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_58.3: type = converted %int.make_type_signed, %.loc4_58.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_61.1: type = struct_type {.i: %T, .different: %i32} [symbolic = %.loc4_61.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongNameStructParam.%.loc4_61.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @WrongNameStructParam.%.loc4_61.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongNameStructParam.decl: %CallWrongNameStructParam.type = fn_decl @CallWrongNameStructParam [template = constants.%CallWrongNameStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongNameStructParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_61.2: type = struct_type {.i: @WrongNameStructParam.%T.loc4_25.2 (%T), .different: %i32} [symbolic = %.loc4_61.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @WrongNameStructParam.%.loc4_61.2 (%.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongNameStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongNameStructParam.ref: %WrongNameStructParam.type = name_ref WrongNameStructParam, file.%WrongNameStructParam.decl [template = constants.%WrongNameStructParam]
// CHECK:STDOUT:   %.loc14_30: Core.IntLiteral = int_value 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_38: Core.IntLiteral = int_value 9 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_39: %.6 = struct_literal (%.loc14_30, %.loc14_38)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongNameStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_61.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_deduce_struct_wrong_order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.first: %T, .second: %i32} [symbolic]
// CHECK:STDOUT:   %WrongOrderStructParam.type: type = fn_type @WrongOrderStructParam [template]
// CHECK:STDOUT:   %WrongOrderStructParam: %WrongOrderStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallWrongOrderStructParam.type: type = fn_type @CallWrongOrderStructParam [template]
// CHECK:STDOUT:   %CallWrongOrderStructParam: %CallWrongOrderStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 11 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.second: Core.IntLiteral, .first: Core.IntLiteral} [template]
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
// CHECK:STDOUT:     .WrongOrderStructParam = %WrongOrderStructParam.decl
// CHECK:STDOUT:     .CallWrongOrderStructParam = %CallWrongOrderStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongOrderStructParam.decl: %WrongOrderStructParam.type = fn_decl @WrongOrderStructParam [template = constants.%WrongOrderStructParam] {
// CHECK:STDOUT:     %T.patt.loc4_26.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_26.1, runtime_param<invalid> [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @WrongOrderStructParam.%.loc4_63.2 (%.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongOrderStructParam.%.loc4_63.2 (%.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_26.1 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_60.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_60.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_60.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_60.3: type = converted %int.make_type_signed, %.loc4_60.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_63.1: type = struct_type {.first: %T, .second: %i32} [symbolic = %.loc4_63.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongOrderStructParam.%.loc4_63.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @WrongOrderStructParam.%.loc4_63.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongOrderStructParam.decl: %CallWrongOrderStructParam.type = fn_decl @CallWrongOrderStructParam [template = constants.%CallWrongOrderStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongOrderStructParam(%T.loc4_26.1: type) {
// CHECK:STDOUT:   %T.loc4_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_63.2: type = struct_type {.first: @WrongOrderStructParam.%T.loc4_26.2 (%T), .second: %i32} [symbolic = %.loc4_63.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @WrongOrderStructParam.%.loc4_63.2 (%.2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongOrderStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongOrderStructParam.ref: %WrongOrderStructParam.type = name_ref WrongOrderStructParam, file.%WrongOrderStructParam.decl [template = constants.%WrongOrderStructParam]
// CHECK:STDOUT:   %.loc14_36: Core.IntLiteral = int_value 11 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_49: Core.IntLiteral = int_value 10 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_51: %.6 = struct_literal (%.loc14_36, %.loc14_49)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongOrderStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %.loc4_63.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplicitNotDeducible = %ImplicitNotDeducible.decl
// CHECK:STDOUT:     .CallImplicitNotDeducible = %CallImplicitNotDeducible.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitNotDeducible.decl: %ImplicitNotDeducible.type = fn_decl @ImplicitNotDeducible [template = constants.%ImplicitNotDeducible] {
// CHECK:STDOUT:     %T.patt.loc6_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_25.1, runtime_param<invalid> [symbolic = %T.patt.loc6_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc6_35.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_35.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc6_35.1, runtime_param<invalid> [symbolic = %U.patt.loc6_35.2 (constants.%U.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @ImplicitNotDeducible.%U.loc6_35.2 (%U) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitNotDeducible.%U.loc6_35.2 (%U) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_25.1 [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc6_35.1 [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6_35.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:     %x.param: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.loc6_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @ImplicitNotDeducible.%U.loc6_35.2 (%U) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%U.loc6_35.2 (%U) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [template = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc6_25.1: type, %U.loc6_35.1: type) {
// CHECK:STDOUT:   %T.loc6_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc6_35.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc6_35.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc6_35.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc6_35.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: type](%x.param_patt: @ImplicitNotDeducible.%T.loc6_25.2 (%T)) -> @ImplicitNotDeducible.%U.loc6_35.2 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [template = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %.loc16: Core.IntLiteral = int_value 42 [template = constants.%.1]
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
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImplicitNotDeducible = %ImplicitNotDeducible.decl
// CHECK:STDOUT:     .CallImplicitNotDeducible = %CallImplicitNotDeducible.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ImplicitNotDeducible.decl: %ImplicitNotDeducible.type = fn_decl @ImplicitNotDeducible [template = constants.%ImplicitNotDeducible] {
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %y.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = binding_pattern y
// CHECK:STDOUT:     %y.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param_pattern %y.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_50: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %y.param: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = value_param runtime_param1
// CHECK:STDOUT:     %y: @ImplicitNotDeducible.%T.loc4_25.2 (%T) = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref @ImplicitNotDeducible.%T.loc4_25.2 (%T) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%T.loc4_25.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [template = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T), %y.param_patt: @ImplicitNotDeducible.%T.loc4_25.2 (%T)) -> @ImplicitNotDeducible.%T.loc4_25.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [template = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %.loc13_24: Core.IntLiteral = int_value 42 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_34: Core.IntLiteral = int_value 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_36: %.3 = struct_literal (%.loc13_34)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
