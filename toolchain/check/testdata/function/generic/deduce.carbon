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

// --- fail_todo_deduce_nested_struct.carbon

library "[[@TEST_NAME]]";

fn StructParam[T:! type](x: {.a: T, .b: i32}) {}

fn CallStructParam() {
  // CHECK:STDERR: fail_todo_deduce_nested_struct.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   StructParam({.a = 1, .b = 2});
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_nested_struct.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn StructParam[T:! type](x: {.a: T, .b: i32}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  StructParam({.a = 1, .b = 2});
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
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallExplicitGenericParam.type: type = fn_type @CallExplicitGenericParam [template]
// CHECK:STDOUT:   %CallExplicitGenericParam: %CallExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(i32) [template]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.6: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.type: type = fn_type @CallExplicitGenericParamWithGenericArg [template]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg: %CallExplicitGenericParamWithGenericArg.type = struct_value () [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%.6) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
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
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_37.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_37.2: type = converted %int.make_type_32.loc6, %.loc6_37.1 [template = i32]
// CHECK:STDOUT:     %.loc6_37.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.decl: %CallExplicitGenericParamWithGenericArg.type = fn_decl @CallExplicitGenericParamWithGenericArg [template = constants.%CallExplicitGenericParamWithGenericArg] {
// CHECK:STDOUT:     %T.patt.loc10_43.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc10_43.1, runtime_param<invalid> [symbolic = %T.patt.loc10_43.2 (constants.%T.patt.2)]
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
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   %.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_50.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @ExplicitGenericParam.%.loc4_39.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc4_50.2 (constants.%.3)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%.loc4_39.2 (%.1) = call %.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%.loc4_39.2 (%.1) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%.loc4_39.2 (%.1) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitGenericParam() -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_30.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_30.2: type = converted %int.make_type_32.loc7, %.loc7_30.1 [template = i32]
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(i32) [template = constants.%.5]
// CHECK:STDOUT:   %ExplicitGenericParam.call: init %.4 = call %.loc7_10()
// CHECK:STDOUT:   %.loc7_35.1: %.4 = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_35.2: %.4 = converted %ExplicitGenericParam.call, %.loc7_35.1
// CHECK:STDOUT:   return %.loc7_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamWithGenericArg(%T.loc10_43.1: type) {
// CHECK:STDOUT:   %T.loc10_43.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_43.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc10_43.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc10_43.2 (constants.%T.patt.2)]
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
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.3
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
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <specific function> = specific_function %ExplicitGenericParam, @ExplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_25.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_25.1, runtime_param<invalid> [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
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
// CHECK:STDOUT:     %T.patt.loc6_34.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt.2)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_34.1, runtime_param<invalid> [symbolic = %T.patt.loc6_34.2 (constants.%T.patt.2)]
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
// CHECK:STDOUT:   %T.patt.loc4_25.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_25.2 (constants.%T.patt.1)]
// CHECK:STDOUT:   %.loc4_39.2: type = ptr_type @ExplicitGenericParam.%T.loc4_25.2 (%T) [symbolic = %.loc4_39.2 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc4_50.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_50.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) -> @ExplicitGenericParam.%.loc4_39.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc4_71: type = name_ref T, %T.loc4_25.1 [symbolic = %T.loc4_25.2 (constants.%T)]
// CHECK:STDOUT:     %.loc4_50.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc4_50.2 (constants.%.3)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @ExplicitGenericParam.%.loc4_39.2 (%.1) = call %.loc4_50.1()
// CHECK:STDOUT:     %.loc4_73.1: @ExplicitGenericParam.%.loc4_39.2 (%.1) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_73.2: @ExplicitGenericParam.%.loc4_39.2 (%.1) = converted %ExplicitGenericParam.call, %.loc4_73.1
// CHECK:STDOUT:     return %.loc4_73.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamConst(%T.loc6_34.1: type) {
// CHECK:STDOUT:   %T.loc6_34.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_34.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_34.2 (constants.%T.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_3.2: <specific function> = specific_function constants.%ExplicitGenericParam, @ExplicitGenericParam(%T.loc6_34.2) [symbolic = %.loc7_3.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc7_23: type = ptr_type @CallExplicitGenericParamConst.%T.loc6_34.2 (%T) [symbolic = %.loc7_23 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_34.1 [symbolic = %T.loc6_34.2 (constants.%T)]
// CHECK:STDOUT:     %.loc7_3.1: <specific function> = specific_function %ExplicitGenericParam.ref, @ExplicitGenericParam(constants.%T) [symbolic = %.loc7_3.2 (constants.%.3)]
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
// CHECK:STDOUT:   %.loc4_50.2 => constants.%.3
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
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced: %ExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T) [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %A [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.type: type = fn_type @CallExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced: %CallExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%A) [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.1 [template]
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
// CHECK:STDOUT:     %return.patt: %.6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.6 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc10: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc10_37: type = ptr_type %A [template = constants.%.6]
// CHECK:STDOUT:     %return.param: ref %.6 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.6 = return_slot %return.param
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
// CHECK:STDOUT:   %.loc7_10.2: <specific function> = specific_function constants.%ExplicitAndAlsoDeduced, @ExplicitAndAlsoDeduced(%T.loc6_27.2) [symbolic = %.loc7_10.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: type, %x.param_patt: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T)) -> @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [template = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc6_27.1 [symbolic = %T.loc6_27.2 (constants.%T)]
// CHECK:STDOUT:     %x.ref: @ExplicitAndAlsoDeduced.%T.loc6_27.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc7_10.1: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%T) [symbolic = %.loc7_10.2 (constants.%.5)]
// CHECK:STDOUT:     %ExplicitAndAlsoDeduced.call: init @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = call %.loc7_10.1(%x.ref)
// CHECK:STDOUT:     %.loc7_38.1: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:     %.loc7_38.2: @ExplicitAndAlsoDeduced.%.loc6_47.2 (%.3) = converted %ExplicitAndAlsoDeduced.call, %.loc7_38.1
// CHECK:STDOUT:     return %.loc7_38.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitAndAlsoDeduced() -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [template = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:   %A.ref.loc11: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc11_37.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc11_10: <specific function> = specific_function %ExplicitAndAlsoDeduced.ref, @ExplicitAndAlsoDeduced(constants.%A) [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_37.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc11_37.3: init %A = class_init (), %.loc11_37.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_37.4: ref %A = temporary %.loc11_37.2, %.loc11_37.3
// CHECK:STDOUT:   %.loc11_37.5: ref %A = converted %.loc11_37.1, %.loc11_37.4
// CHECK:STDOUT:   %.loc11_37.6: %A = bind_value %.loc11_37.5
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.call: init %.6 = call %.loc11_10(%.loc11_37.6)
// CHECK:STDOUT:   %.loc11_39.1: %.6 = value_of_initializer %ExplicitAndAlsoDeduced.call
// CHECK:STDOUT:   %.loc11_39.2: %.6 = converted %ExplicitAndAlsoDeduced.call, %.loc11_39.1
// CHECK:STDOUT:   return %.loc11_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%T) {
// CHECK:STDOUT:   %T.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_27.2 => constants.%T
// CHECK:STDOUT:   %.loc6_47.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_10.2 => constants.%.5
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
// CHECK:STDOUT:   %.loc6_47.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_10.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ImplicitGenericParam.type: type = fn_type @ImplicitGenericParam [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitGenericParam: %ImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(%T) [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallImplicitGenericParam.type: type = fn_type @CallImplicitGenericParam [template]
// CHECK:STDOUT:   %CallImplicitGenericParam: %CallImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %ImplicitGenericParam, @ImplicitGenericParam(i32) [template]
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
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_32.1: type = value_of_initializer %int.make_type_32.loc6_32 [template = i32]
// CHECK:STDOUT:     %.loc6_32.2: type = converted %int.make_type_32.loc6_32, %.loc6_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_32.loc6_40 [template = i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_32.loc6_40, %.loc6_43.1 [template = i32]
// CHECK:STDOUT:     %.loc6_43.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:     %n.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %n.param
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
// CHECK:STDOUT:   %.loc4_56.2: <specific function> = specific_function constants.%ImplicitGenericParam, @ImplicitGenericParam(%T.loc4_25.2) [symbolic = %.loc4_56.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @ImplicitGenericParam.%T.loc4_25.2 (%T)) -> @ImplicitGenericParam.%.loc4_45.2 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:     %x.ref: @ImplicitGenericParam.%T.loc4_25.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_56.1: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(constants.%T) [symbolic = %.loc4_56.2 (constants.%.3)]
// CHECK:STDOUT:     %ImplicitGenericParam.call: init @ImplicitGenericParam.%.loc4_45.2 (%.1) = call %.loc4_56.1(%x.ref)
// CHECK:STDOUT:     %.loc4_79.1: @ImplicitGenericParam.%.loc4_45.2 (%.1) = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:     %.loc4_79.2: @ImplicitGenericParam.%.loc4_45.2 (%.1) = converted %ImplicitGenericParam.call, %.loc4_79.1
// CHECK:STDOUT:     return %.loc4_79.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitGenericParam(%n.param_patt: i32) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc7_10: <specific function> = specific_function %ImplicitGenericParam.ref, @ImplicitGenericParam(i32) [template = constants.%.5]
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
// CHECK:STDOUT:   %.loc4_56.2 => constants.%.3
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%T, i32) [symbolic]
// CHECK:STDOUT:   %TupleParam.type: type = fn_type @TupleParam [template]
// CHECK:STDOUT:   %TupleParam: %TupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %CallTupleParam.type: type = fn_type @CallTupleParam [template]
// CHECK:STDOUT:   %CallTupleParam: %CallTupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %TupleParam, @TupleParam(i32) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %tuple: %.7 = tuple_value (%.5, %.6) [template]
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
// CHECK:STDOUT:     .TupleParam = %TupleParam.decl
// CHECK:STDOUT:     .CallTupleParam = %CallTupleParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleParam.decl: %TupleParam.type = fn_decl @TupleParam [template = constants.%TupleParam] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_15.1, runtime_param<invalid> [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @TupleParam.%.loc4_35.5 (%.3) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @TupleParam.%.loc4_35.5 (%.3) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_15.1 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_35.1: %.2 = tuple_literal (%T.ref, %int.make_type_32)
// CHECK:STDOUT:     %.loc4_35.2: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_35.3: type = converted %int.make_type_32, %.loc4_35.2 [template = i32]
// CHECK:STDOUT:     %.loc4_35.4: type = converted %.loc4_35.1, constants.%.3 [symbolic = %.loc4_35.5 (constants.%.3)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @TupleParam.%.loc4_35.5 (%.3) = value_param runtime_param0
// CHECK:STDOUT:     %x: @TupleParam.%.loc4_35.5 (%.3) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTupleParam.decl: %CallTupleParam.type = fn_decl @CallTupleParam [template = constants.%CallTupleParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TupleParam(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_35.5: type = tuple_type (@TupleParam.%T.loc4_15.2 (%T), i32) [symbolic = %.loc4_35.5 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%x.param_patt: @TupleParam.%.loc4_35.5 (%.3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTupleParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TupleParam.ref: %TupleParam.type = name_ref TupleParam, file.%TupleParam.decl [template = constants.%TupleParam]
// CHECK:STDOUT:   %.loc7_15: i32 = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_18: i32 = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_19.1: %.7 = tuple_literal (%.loc7_15, %.loc7_18)
// CHECK:STDOUT:   %.loc7_3: <specific function> = specific_function %TupleParam.ref, @TupleParam(i32) [template = constants.%.8]
// CHECK:STDOUT:   %tuple: %.7 = tuple_value (%.loc7_15, %.loc7_18) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_19.2: %.7 = converted %.loc7_19.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %TupleParam.call: init %.1 = call %.loc7_3(%.loc7_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %.loc4_35.5 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(i32) {
// CHECK:STDOUT:   %T.loc4_15.2 => i32
// CHECK:STDOUT:   %T.patt.loc4_15.2 => i32
// CHECK:STDOUT:   %.loc4_35.5 => constants.%.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_deduce_nested_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %T, .b: i32} [symbolic]
// CHECK:STDOUT:   %StructParam.type: type = fn_type @StructParam [template]
// CHECK:STDOUT:   %StructParam: %StructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [symbolic]
// CHECK:STDOUT:   %CallStructParam.type: type = fn_type @CallStructParam [template]
// CHECK:STDOUT:   %CallStructParam: %CallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: i32, .b: i32} [template]
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
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_41.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_41.2: type = converted %int.make_type_32, %.loc4_41.1 [template = i32]
// CHECK:STDOUT:     %.loc4_44.1: type = struct_type {.a: %T, .b: i32} [symbolic = %.loc4_44.2 (constants.%.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @StructParam.%.loc4_44.2 (%.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @StructParam.%.loc4_44.2 (%.2) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallStructParam.decl: %CallStructParam.type = fn_decl @CallStructParam [template = constants.%CallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StructParam(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc4_44.2: type = struct_type {.a: @StructParam.%T.loc4_16.2 (%T), .b: i32} [symbolic = %.loc4_44.2 (constants.%.2)]
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
// CHECK:STDOUT:   %.loc14_21: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_29: i32 = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_30: %.6 = struct_literal (%.loc14_21, %.loc14_29)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %.loc4_44.2 => constants.%.2
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 42 [template]
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
// CHECK:STDOUT:   %.loc16: i32 = int_value 42 [template = constants.%.2]
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_value 42 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 12 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.x: i32} [template]
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
// CHECK:STDOUT:   %.loc13_24: i32 = int_value 42 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_34: i32 = int_value 12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_36: %.4 = struct_literal (%.loc13_34)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_25.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
