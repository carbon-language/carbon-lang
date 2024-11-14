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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallExplicitGenericParam.type: type = fn_type @CallExplicitGenericParam [template]
// CHECK:STDOUT:   %CallExplicitGenericParam: %CallExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(i32) [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.type: type = fn_type @CallExplicitGenericParamWithGenericArg [template]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg: %CallExplicitGenericParamWithGenericArg.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%.5) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_37.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_37.2: type = converted %int.make_type_32.loc6, %.loc6_37.1 [template = i32]
// CHECK:STDOUT:     %.loc6_37.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.decl: %CallExplicitGenericParamWithGenericArg.type = fn_decl @CallExplicitGenericParamWithGenericArg [template = constants.%CallExplicitGenericParamWithGenericArg] {
// CHECK:STDOUT:     %T.patt.loc10_43.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_43.1, runtime_param<invalid> [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %.loc10_62.1: type = struct_type {.a: %T} [symbolic = %.loc10_62.2 (constants.%.5)]
// CHECK:STDOUT:     %.loc10_63.1: type = ptr_type %.5 [symbolic = %.loc10_63.2 (constants.%.6)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10_43.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = return_slot %return.param
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParam() -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_32.loc7, %.loc7_30.1 [template = i32]
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(i32) [template = constants.%.4]
// CHECK:STDOUT:   %ExplicitGenericParam.call: init %.3 = call %.loc7_10()
// CHECK:STDOUT:   %.loc7_35.1: %.3 = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_35.2: %.3 = converted %ExplicitGenericParam.call, %.loc7_35.1
// CHECK:STDOUT:   return %.loc7_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamWithGenericArg(%T.loc10_43.1: type) {
// CHECK:STDOUT:   %T.loc10_43.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_43.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc10_62.2: type = struct_type {.a: @CallExplicitGenericParamWithGenericArg.%T.loc10_43.2 (%T)} [symbolic = %.loc10_62.2 (constants.%.5)]
// CHECK:STDOUT:   %.loc10_63.2: type = ptr_type @CallExplicitGenericParamWithGenericArg.%.loc10_62.2 (%.5) [symbolic = %.loc10_63.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc11_10.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%.loc10_62.2) [symbolic = %.loc11_10.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc11: type = name_ref T, %T.loc10_43.1 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:     %.loc11_37: type = struct_type {.a: %T} [symbolic = %.loc10_62.2 (constants.%.5)]
// CHECK:STDOUT:     %.loc11_10.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%.5) [symbolic = %.loc11_10.2 (constants.%.7)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = call %.loc11_10.1()
// CHECK:STDOUT:     %.loc11_39.1: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc11_39.2: @CallExplicitGenericParamWithGenericArg.%.loc10_63.2 (%.6) = converted %ExplicitGenericParam.call, %.loc11_39.1
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
// CHECK:STDOUT: specific @ExplicitGenericParam(@ExplicitGenericParam.%T.loc4_25.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => i32
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamWithGenericArg(constants.%T) {
// CHECK:STDOUT:   %T.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc10_43.2 => constants.%T
// CHECK:STDOUT:   %.loc10_62.2 => constants.%.5
// CHECK:STDOUT:   %.loc10_63.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%.5) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%.5
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%.5
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(@CallExplicitGenericParamWithGenericArg.%.loc10_62.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%.5
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%.5
// CHECK:STDOUT:   %.loc4_39.2 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_explicit_non_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
// CHECK:STDOUT: specific @ExplicitGenericParam(@ExplicitGenericParam.%T.loc4_25.2) {
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced: %ExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %A [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.type: type = fn_type @CallExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced: %CallExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.6: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%A) [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.1 [template]
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
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(@ExplicitAndAlsoDeduced.%T.loc6_27.2) {
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitGenericParam: %ImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallImplicitGenericParam.type: type = fn_type @CallImplicitGenericParam [template]
// CHECK:STDOUT:   %CallImplicitGenericParam: %CallImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_32.1: type = value_of_initializer %int.make_type_32.loc6_32 [template = i32]
// CHECK:STDOUT:     %.loc6_32.2: type = converted %int.make_type_32.loc6_32, %.loc6_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_32.loc6_40 [template = i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_32.loc6_40, %.loc6_43.1 [template = i32]
// CHECK:STDOUT:     %.loc6_43.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitGenericParam(%n.param_patt: i32) -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(i32) [template = constants.%.4]
// CHECK:STDOUT:   %ImplicitGenericParam.call: init %.3 = call %.loc7_10(%n.ref)
// CHECK:STDOUT:   %.loc7_33.1: %.3 = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_33.2: %.3 = converted %ImplicitGenericParam.call, %.loc7_33.1
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
// CHECK:STDOUT: specific @ImplicitGenericParam(@ImplicitGenericParam.%T.loc4_25.2) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_45.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(i32) {
// CHECK:STDOUT:   %T.loc4_25.2 => i32
// CHECK:STDOUT:   %T.patt.loc4_25.2 => i32
// CHECK:STDOUT:   %.loc4_45.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%T, i32) [symbolic]
// CHECK:STDOUT:   %TupleParam.type: type = fn_type @TupleParam [template]
// CHECK:STDOUT:   %TupleParam: %TupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type %tuple.type.2 [symbolic]
// CHECK:STDOUT:   %CallTupleParam.type: type = fn_type @CallTupleParam [template]
// CHECK:STDOUT:   %CallTupleParam: %CallTupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (Core.IntLiteral, i32) [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %TupleParam, @TupleParam(Core.IntLiteral) [template]
// CHECK:STDOUT:   %.5: type = ptr_type %tuple.type.4 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.10: %.6 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.11: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.11)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.12: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.13: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.13)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.14: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.15: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.16: %.15 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.18: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.19: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.20: %.19 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.11)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.13)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.25: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.26: %.25 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.4 = tuple_value (%.2, %.31) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.6) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.29]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.26 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.32 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.11)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
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
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_35.1: %tuple.type.1 = tuple_literal (%T.ref, %int.make_type_32)
// CHECK:STDOUT:     %.loc4_35.2: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_35.3: type = converted %int.make_type_32, %.loc4_35.2 [template = i32]
// CHECK:STDOUT:     %.loc4_35.4: type = converted %.loc4_35.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TupleParam.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @TupleParam.%tuple.type (%tuple.type.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTupleParam.decl: %CallTupleParam.type = fn_decl @CallTupleParam [template = constants.%CallTupleParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.6) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.19)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.19) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TupleParam(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@TupleParam.%T.loc4_15.2 (%T), i32) [symbolic = %tuple.type (constants.%tuple.type.2)]
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
// CHECK:STDOUT:   %.loc7_15: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_18: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_19.1: %tuple.type.3 = tuple_literal (%.loc7_15, %.loc7_18)
// CHECK:STDOUT:   %.loc7_3: <specific function> = specific_function %TupleParam.ref, @TupleParam(Core.IntLiteral) [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_19.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc7_19.3: <bound method> = bound_method %.loc7_18, %.loc7_19.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc7_19.3(%.loc7_18) [template = constants.%.31]
// CHECK:STDOUT:   %.loc7_19.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:   %.loc7_19.5: i32 = converted %.loc7_18, %.loc7_19.4 [template = constants.%.31]
// CHECK:STDOUT:   %tuple: %tuple.type.4 = tuple_value (%.loc7_15, %.loc7_19.5) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_19.6: %tuple.type.4 = converted %.loc7_19.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %TupleParam.call: init %empty_tuple.type = call %.loc7_3(%.loc7_19.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.11) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.13) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.11)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.13)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.4)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.11) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.13) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.11)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.13)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
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
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.11) {
// CHECK:STDOUT:   %Dest => constants.%.11
// CHECK:STDOUT:   %Dest.patt => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.11
// CHECK:STDOUT:   %Dest.patt => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.13) {
// CHECK:STDOUT:   %Dest => constants.%.13
// CHECK:STDOUT:   %Dest.patt => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.13
// CHECK:STDOUT:   %Dest.patt => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.15
// CHECK:STDOUT:   %.2 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.9
// CHECK:STDOUT:   %.1 => constants.%.21
// CHECK:STDOUT:   %.2 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.11) {
// CHECK:STDOUT:   %Dest => constants.%.11
// CHECK:STDOUT:   %Dest.patt => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.11
// CHECK:STDOUT:   %Dest.patt => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.13) {
// CHECK:STDOUT:   %Dest => constants.%.13
// CHECK:STDOUT:   %Dest.patt => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.13
// CHECK:STDOUT:   %Dest.patt => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.25
// CHECK:STDOUT:   %.2 => constants.%.26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_nested_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: %T, .b: i32} [symbolic]
// CHECK:STDOUT:   %StructParam.type: type = fn_type @StructParam [template]
// CHECK:STDOUT:   %StructParam: %StructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %CallStructParam.type: type = fn_type @CallStructParam [template]
// CHECK:STDOUT:   %CallStructParam: %CallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: Core.IntLiteral, .b: i32} [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %StructParam, @StructParam(Core.IntLiteral) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.9: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.10: %.9 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.13: %.9 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %.14: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(%.14)> [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.2(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.15: <witness> = interface_witness (%Convert.3) [symbolic]
// CHECK:STDOUT:   %.16: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5: type = facet_type <@ImplicitAs, @ImplicitAs(%.16)> [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.3, @impl.3(%N) [symbolic]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [symbolic]
// CHECK:STDOUT:   %.17: <witness> = interface_witness (%Convert.4) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.18: type = assoc_entity_type %ImplicitAs.type.6, %Convert.type.5 [template]
// CHECK:STDOUT:   %.19: %.18 = assoc_entity element0, imports.%import_ref.20 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.4, @impl.5(%N) [symbolic]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.6) [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5, @impl.6(%N) [symbolic]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %.21: <witness> = interface_witness (%Convert.7) [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.3: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(i32)> [template]
// CHECK:STDOUT:   %Self.4: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.6, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %.22: type = assoc_entity_type %As.type.2, %Convert.type.8 [symbolic]
// CHECK:STDOUT:   %.23: %.22 = assoc_entity element0, imports.%import_ref.33 [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %.24: type = assoc_entity_type %As.type.3, %Convert.type.9 [template]
// CHECK:STDOUT:   %.25: %.24 = assoc_entity element0, imports.%import_ref.34 [template]
// CHECK:STDOUT:   %As.type.4: type = facet_type <@As, @As(%.14)> [symbolic]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.7, @impl.8(%N) [symbolic]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [symbolic]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.10) [symbolic]
// CHECK:STDOUT:   %As.type.5: type = facet_type <@As, @As(%.16)> [symbolic]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.8, @impl.9(%N) [symbolic]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [symbolic]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.11) [symbolic]
// CHECK:STDOUT:   %As.type.6: type = facet_type <@As, @As(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.12: type = fn_type @Convert.6, @As(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.12: %Convert.type.12 = struct_value () [template]
// CHECK:STDOUT:   %.28: type = assoc_entity_type %As.type.6, %Convert.type.12 [template]
// CHECK:STDOUT:   %.29: %.28 = assoc_entity element0, imports.%import_ref.44 [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.9, @impl.11(%N) [symbolic]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [symbolic]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.13) [symbolic]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.10, @impl.12(%N) [symbolic]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [symbolic]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [symbolic]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.6 = struct_value (%.3, %.34) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+55, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.4: @ImplicitAs.%.1 (%.9) = import_ref Core//prelude/operators/as, inst+83, loaded [symbolic = @ImplicitAs.%.2 (constants.%.13)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//prelude/operators/as, inst+100, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.9: type = import_ref Core//prelude/operators/as, inst+111, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.10: <witness> = import_ref Core//prelude/operators/as, inst+133, loaded [template = constants.%.32]
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//prelude/operators/as, inst+147, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//prelude/operators/as, inst+165, loaded [symbolic = @impl.2.%ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:   %import_ref.13 = import_ref Core//prelude/operators/as, inst+196, unloaded
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//prelude/operators/as, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//prelude/operators/as, inst+229, loaded [symbolic = @impl.3.%ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//prelude/operators/as, inst+260, unloaded
// CHECK:STDOUT:   %import_ref.17: type = import_ref Core//prelude/operators/as, inst+267, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.18: type = import_ref Core//prelude/operators/as, inst+273, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//prelude/operators/as, inst+296, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Core//prelude/operators/as, inst+76, unloaded
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//prelude/operators/as, inst+309, loaded [symbolic = @impl.5.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.22: type = import_ref Core//prelude/operators/as, inst+315, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//prelude/operators/as, inst+339, unloaded
// CHECK:STDOUT:   %import_ref.24: type = import_ref Core//prelude/operators/as, inst+355, loaded [symbolic = @impl.6.%.1 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//prelude/operators/as, inst+361, loaded [template = constants.%ImplicitAs.type.6]
// CHECK:STDOUT:   %import_ref.26 = import_ref Core//prelude/operators/as, inst+385, unloaded
// CHECK:STDOUT:   %import_ref.27 = import_ref Core//prelude/operators/as, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.28 = import_ref Core//prelude/operators/as, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.29 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.30: type = import_ref Core//prelude/operators/as, inst+393, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.31: type = import_ref Core//prelude/operators/as, inst+398, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.32 = import_ref Core//prelude/operators/as, inst+420, unloaded
// CHECK:STDOUT:   %import_ref.33 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.34 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.35: type = import_ref Core//prelude/operators/as, inst+432, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.36: type = import_ref Core//prelude/operators/as, inst+439, loaded [symbolic = @impl.8.%As.type (constants.%As.type.4)]
// CHECK:STDOUT:   %import_ref.37 = import_ref Core//prelude/operators/as, inst+470, unloaded
// CHECK:STDOUT:   %import_ref.38: type = import_ref Core//prelude/operators/as, inst+485, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.39: type = import_ref Core//prelude/operators/as, inst+492, loaded [symbolic = @impl.9.%As.type (constants.%As.type.5)]
// CHECK:STDOUT:   %import_ref.40 = import_ref Core//prelude/operators/as, inst+523, unloaded
// CHECK:STDOUT:   %import_ref.41: type = import_ref Core//prelude/operators/as, inst+530, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.42: type = import_ref Core//prelude/operators/as, inst+536, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.43 = import_ref Core//prelude/operators/as, inst+559, unloaded
// CHECK:STDOUT:   %import_ref.44 = import_ref Core//prelude/operators/as, inst+36, unloaded
// CHECK:STDOUT:   %import_ref.45: type = import_ref Core//prelude/operators/as, inst+572, loaded [symbolic = @impl.11.%.1 (constants.%.14)]
// CHECK:STDOUT:   %import_ref.46: type = import_ref Core//prelude/operators/as, inst+578, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.47 = import_ref Core//prelude/operators/as, inst+602, unloaded
// CHECK:STDOUT:   %import_ref.48: type = import_ref Core//prelude/operators/as, inst+618, loaded [symbolic = @impl.12.%.1 (constants.%.16)]
// CHECK:STDOUT:   %import_ref.49: type = import_ref Core//prelude/operators/as, inst+624, loaded [template = constants.%As.type.6]
// CHECK:STDOUT:   %import_ref.50 = import_ref Core//prelude/operators/as, inst+648, unloaded
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
// CHECK:STDOUT:     %x.patt: @StructParam.%.loc4_44.2 (%.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @StructParam.%.loc4_44.2 (%.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.1 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_41.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_41.2: type = converted %int.make_type_32, %.loc4_41.1 [template = i32]
// CHECK:STDOUT:     %.loc4_44.1: type = struct_type {.a: %T, .b: i32} [symbolic = %.loc4_44.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @StructParam.%.loc4_44.2 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @StructParam.%.loc4_44.2 (%.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallStructParam.decl: %CallStructParam.type = fn_decl @CallStructParam [template = constants.%CallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.9)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.9) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %Dest.patt: type = symbolic_binding_pattern Dest, 0 [symbolic = %Dest.patt (constants.%Dest.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.6, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.8)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.8) = struct_value () [symbolic = %Convert (constants.%Convert.8)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.8) [symbolic = %.1 (constants.%.22)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.22) = assoc_entity element0, imports.%import_ref.33 [symbolic = %.2 (constants.%.23)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.27
// CHECK:STDOUT:     .Convert = imports.%import_ref.28
// CHECK:STDOUT:     witness = (imports.%import_ref.29)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.8 as imports.%import_ref.9 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @impl.2(%N) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @impl.2.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.15)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.3, @impl.3(%N) [symbolic = %Convert.type (constants.%Convert.type.4)]
// CHECK:STDOUT:   %Convert: @impl.3.%Convert.type (%Convert.type.4) = struct_value () [symbolic = %Convert (constants.%Convert.4)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.17 as imports.%import_ref.18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.4, @impl.5(%N) [symbolic = %Convert.type (constants.%Convert.type.6)]
// CHECK:STDOUT:   %Convert: @impl.5.%Convert.type (%Convert.type.6) = struct_value () [symbolic = %Convert (constants.%Convert.6)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.21 as imports.%import_ref.22 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.23
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.6(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.5, @impl.6(%N) [symbolic = %Convert.type (constants.%Convert.type.7)]
// CHECK:STDOUT:   %Convert: @impl.6.%Convert.type (%Convert.type.7) = struct_value () [symbolic = %Convert (constants.%Convert.7)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.21)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.24 as imports.%import_ref.25 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: imports.%import_ref.30 as imports.%import_ref.31 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.7, @impl.8(%N) [symbolic = %Convert.type (constants.%Convert.type.10)]
// CHECK:STDOUT:   %Convert: @impl.8.%Convert.type (%Convert.type.10) = struct_value () [symbolic = %Convert (constants.%Convert.10)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.26)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.35 as imports.%import_ref.36 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.37
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%.1)> [symbolic = %As.type (constants.%As.type.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.8, @impl.9(%N) [symbolic = %Convert.type (constants.%Convert.type.11)]
// CHECK:STDOUT:   %Convert: @impl.9.%Convert.type (%Convert.type.11) = struct_value () [symbolic = %Convert (constants.%Convert.11)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.38 as imports.%import_ref.39 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.40
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: imports.%import_ref.41 as imports.%import_ref.42 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.11(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.9, @impl.11(%N) [symbolic = %Convert.type (constants.%Convert.type.13)]
// CHECK:STDOUT:   %Convert: @impl.11.%Convert.type (%Convert.type.13) = struct_value () [symbolic = %Convert (constants.%Convert.13)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.30)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.45 as imports.%import_ref.46 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl.12(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.10, @impl.12(%N) [symbolic = %Convert.type (constants.%Convert.type.14)]
// CHECK:STDOUT:   %Convert: @impl.12.%Convert.type (%Convert.type.14) = struct_value () [symbolic = %Convert (constants.%Convert.14)]
// CHECK:STDOUT:   %.2: <witness> = interface_witness (%Convert) [symbolic = %.2 (constants.%.31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%import_ref.48 as imports.%import_ref.49 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%import_ref.50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StructParam(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_44.2: type = struct_type {.a: @StructParam.%T.loc4_16.2 (%T), .b: i32} [symbolic = %.loc4_44.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @StructParam.%.loc4_44.2 (%.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %StructParam.ref: %StructParam.type = name_ref StructParam, file.%StructParam.decl [template = constants.%StructParam]
// CHECK:STDOUT:   %.loc7_21: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_29: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_30.1: %.5 = struct_literal (%.loc7_21, %.loc7_29)
// CHECK:STDOUT:   %.loc7_3: <specific function> = specific_function %StructParam.ref, @StructParam(Core.IntLiteral) [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_30.2: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc7_30.3: <bound method> = bound_method %.loc7_29, %.loc7_30.2 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc7_30.3(%.loc7_29) [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_30.4: i32 = value_of_initializer %int.convert_checked [template = constants.%.34]
// CHECK:STDOUT:   %.loc7_30.5: i32 = converted %.loc7_29, %.loc7_30.4 [template = constants.%.34]
// CHECK:STDOUT:   %struct: %.6 = struct_value (%.loc7_21, %.loc7_30.5) [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_30.6: %.6 = converted %.loc7_30.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %StructParam.call: init %empty_tuple.type = call %.loc7_3(%.loc7_30.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.2.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.3(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.3.%.1 (%.16) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.4(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.4.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.5(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.5.%.1 (%.16)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.6(constants.%Dest: type, constants.%Self.3: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%Dest)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.6.%Self (%Self.4)]() -> @Convert.6.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.7(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.7.%.1 (%.14) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.8(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: Core.IntLiteral]() -> @Convert.8.%.1 (%.16) = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.9(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type signed, %N [symbolic = %.1 (constants.%.14)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.9.%.1 (%.14)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.10(constants.%N: Core.IntLiteral) {
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N (constants.%N)]
// CHECK:STDOUT:   %.1: type = int_type unsigned, %N [symbolic = %.1 (constants.%.16)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.10.%.1 (%.16)]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.11[%self.param_patt: Core.IntLiteral]() -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %.loc4_44.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc4_16.2 => Core.IntLiteral
// CHECK:STDOUT:   %.loc4_44.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.11
// CHECK:STDOUT:   %.2 => constants.%.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.2.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.2(@impl.2.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%.16) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@impl.3.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.3(@impl.3.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.3(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.6
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.18
// CHECK:STDOUT:   %.2 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.5(@impl.5.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.4(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.6(@impl.6.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.5(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(i32) {
// CHECK:STDOUT:   %Dest => i32
// CHECK:STDOUT:   %Dest.patt => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.9
// CHECK:STDOUT:   %Convert => constants.%Convert.9
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT:   %.2 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.6.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %Dest.patt => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.6(constants.%Dest, constants.%Self.3) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.14) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.8.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.14
// CHECK:STDOUT:   %Dest.patt => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.8(@impl.8.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %As.type => constants.%As.type.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.7(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%.16) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@impl.9.%.1) {
// CHECK:STDOUT:   %Dest => constants.%.16
// CHECK:STDOUT:   %Dest.patt => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.9(@impl.9.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %As.type => constants.%As.type.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.8(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(Core.IntLiteral) {
// CHECK:STDOUT:   %Dest => Core.IntLiteral
// CHECK:STDOUT:   %Dest.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.6
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.12
// CHECK:STDOUT:   %Convert => constants.%Convert.12
// CHECK:STDOUT:   %.1 => constants.%.28
// CHECK:STDOUT:   %.2 => constants.%.29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.11(@impl.11.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.9(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl.12(@impl.12.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.10(constants.%N) {
// CHECK:STDOUT:   %N => constants.%N
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_bigger_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.c: %T, .d: i32, .e: i32} [symbolic]
// CHECK:STDOUT:   %BigStructParam.type: type = fn_type @BigStructParam [template]
// CHECK:STDOUT:   %BigStructParam: %BigStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %CallBigStructParam.type: type = fn_type @CallBigStructParam [template]
// CHECK:STDOUT:   %CallBigStructParam: %CallBigStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.c: Core.IntLiteral, .d: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %x.patt: @BigStructParam.%.loc4_56.2 (%.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @BigStructParam.%.loc4_56.2 (%.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_19.1 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32.loc4_44: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_44.1: type = value_of_initializer %int.make_type_32.loc4_44 [template = i32]
// CHECK:STDOUT:     %.loc4_44.2: type = converted %int.make_type_32.loc4_44, %.loc4_44.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_53: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_53.1: type = value_of_initializer %int.make_type_32.loc4_53 [template = i32]
// CHECK:STDOUT:     %.loc4_53.2: type = converted %int.make_type_32.loc4_53, %.loc4_53.1 [template = i32]
// CHECK:STDOUT:     %.loc4_56.1: type = struct_type {.c: %T, .d: i32, .e: i32} [symbolic = %.loc4_56.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @BigStructParam.%.loc4_56.2 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @BigStructParam.%.loc4_56.2 (%.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallBigStructParam.decl: %CallBigStructParam.type = fn_decl @CallBigStructParam [template = constants.%CallBigStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BigStructParam(%T.loc4_19.1: type) {
// CHECK:STDOUT:   %T.loc4_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_56.2: type = struct_type {.c: @BigStructParam.%T.loc4_19.2 (%T), .d: i32, .e: i32} [symbolic = %.loc4_56.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @BigStructParam.%.loc4_56.2 (%.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallBigStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BigStructParam.ref: %BigStructParam.type = name_ref BigStructParam, file.%BigStructParam.decl [template = constants.%BigStructParam]
// CHECK:STDOUT:   %.loc14_24: Core.IntLiteral = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_32: Core.IntLiteral = int_value 4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_33: %.5 = struct_literal (%.loc14_24, %.loc14_32)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BigStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_19.2 => constants.%T
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_smaller_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.f: %T, .g: i32} [symbolic]
// CHECK:STDOUT:   %SmallStructParam.type: type = fn_type @SmallStructParam [template]
// CHECK:STDOUT:   %SmallStructParam: %SmallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %CallSmallStructParam.type: type = fn_type @CallSmallStructParam [template]
// CHECK:STDOUT:   %CallSmallStructParam: %CallSmallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.f: Core.IntLiteral, .g: Core.IntLiteral, .h: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %x.patt: @SmallStructParam.%.loc4_49.2 (%.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @SmallStructParam.%.loc4_49.2 (%.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_21.1 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_46.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_46.2: type = converted %int.make_type_32, %.loc4_46.1 [template = i32]
// CHECK:STDOUT:     %.loc4_49.1: type = struct_type {.f: %T, .g: i32} [symbolic = %.loc4_49.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @SmallStructParam.%.loc4_49.2 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @SmallStructParam.%.loc4_49.2 (%.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallSmallStructParam.decl: %CallSmallStructParam.type = fn_decl @CallSmallStructParam [template = constants.%CallSmallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SmallStructParam(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_21.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_21.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_49.2: type = struct_type {.f: @SmallStructParam.%T.loc4_21.2 (%T), .g: i32} [symbolic = %.loc4_49.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @SmallStructParam.%.loc4_49.2 (%.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSmallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SmallStructParam.ref: %SmallStructParam.type = name_ref SmallStructParam, file.%SmallStructParam.decl [template = constants.%SmallStructParam]
// CHECK:STDOUT:   %.loc14_26: Core.IntLiteral = int_value 5 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_34: Core.IntLiteral = int_value 6 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_42: Core.IntLiteral = int_value 7 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_43: %.6 = struct_literal (%.loc14_26, %.loc14_34, %.loc14_42)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SmallStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_21.2 => constants.%T
// CHECK:STDOUT:   %.loc4_49.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_struct_wrong_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.i: %T, .different: i32} [symbolic]
// CHECK:STDOUT:   %WrongNameStructParam.type: type = fn_type @WrongNameStructParam [template]
// CHECK:STDOUT:   %WrongNameStructParam: %WrongNameStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %CallWrongNameStructParam.type: type = fn_type @CallWrongNameStructParam [template]
// CHECK:STDOUT:   %CallWrongNameStructParam: %CallWrongNameStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.i: Core.IntLiteral, .j: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %x.patt: @WrongNameStructParam.%.loc4_61.2 (%.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongNameStructParam.%.loc4_61.2 (%.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_58.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_58.2: type = converted %int.make_type_32, %.loc4_58.1 [template = i32]
// CHECK:STDOUT:     %.loc4_61.1: type = struct_type {.i: %T, .different: i32} [symbolic = %.loc4_61.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_25.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongNameStructParam.%.loc4_61.2 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @WrongNameStructParam.%.loc4_61.2 (%.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongNameStructParam.decl: %CallWrongNameStructParam.type = fn_decl @CallWrongNameStructParam [template = constants.%CallWrongNameStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongNameStructParam(%T.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_25.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_61.2: type = struct_type {.i: @WrongNameStructParam.%T.loc4_25.2 (%T), .different: i32} [symbolic = %.loc4_61.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @WrongNameStructParam.%.loc4_61.2 (%.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongNameStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongNameStructParam.ref: %WrongNameStructParam.type = name_ref WrongNameStructParam, file.%WrongNameStructParam.decl [template = constants.%WrongNameStructParam]
// CHECK:STDOUT:   %.loc14_30: Core.IntLiteral = int_value 8 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_38: Core.IntLiteral = int_value 9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_39: %.5 = struct_literal (%.loc14_30, %.loc14_38)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongNameStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %.loc4_61.2 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_deduce_struct_wrong_order.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.first: %T, .second: i32} [symbolic]
// CHECK:STDOUT:   %WrongOrderStructParam.type: type = fn_type @WrongOrderStructParam [template]
// CHECK:STDOUT:   %WrongOrderStructParam: %WrongOrderStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %CallWrongOrderStructParam.type: type = fn_type @CallWrongOrderStructParam [template]
// CHECK:STDOUT:   %CallWrongOrderStructParam: %CallWrongOrderStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 11 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.second: Core.IntLiteral, .first: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %x.patt: @WrongOrderStructParam.%.loc4_63.2 (%.1) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @WrongOrderStructParam.%.loc4_63.2 (%.1) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_26.1 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_60.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_60.2: type = converted %int.make_type_32, %.loc4_60.1 [template = i32]
// CHECK:STDOUT:     %.loc4_63.1: type = struct_type {.first: %T, .second: i32} [symbolic = %.loc4_63.2 (constants.%.1)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_26.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @WrongOrderStructParam.%.loc4_63.2 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %x: @WrongOrderStructParam.%.loc4_63.2 (%.1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWrongOrderStructParam.decl: %CallWrongOrderStructParam.type = fn_decl @CallWrongOrderStructParam [template = constants.%CallWrongOrderStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongOrderStructParam(%T.loc4_26.1: type) {
// CHECK:STDOUT:   %T.loc4_26.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_26.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_26.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_26.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_63.2: type = struct_type {.first: @WrongOrderStructParam.%T.loc4_26.2 (%T), .second: i32} [symbolic = %.loc4_63.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @WrongOrderStructParam.%.loc4_63.2 (%.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongOrderStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WrongOrderStructParam.ref: %WrongOrderStructParam.type = name_ref WrongOrderStructParam, file.%WrongOrderStructParam.decl [template = constants.%WrongOrderStructParam]
// CHECK:STDOUT:   %.loc14_36: Core.IntLiteral = int_value 11 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_49: Core.IntLiteral = int_value 10 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_51: %.5 = struct_literal (%.loc14_36, %.loc14_49)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongOrderStructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_26.2 => constants.%T
// CHECK:STDOUT:   %.loc4_63.2 => constants.%.1
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
