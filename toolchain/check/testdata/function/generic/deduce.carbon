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

fn ExplicitGenericParam(T:! type) -> T*;

fn CallExplicitGenericParam() -> i32* {
  return ExplicitGenericParam(i32);
}

fn CallExplicitGenericParamWithGenericArg(T:! type) -> {.a: T}* {
  return ExplicitGenericParam({.a: T});
}

// --- fail_todo_explicit_vs_deduced.carbon

library "[[@TEST_NAME]]";

class A {}

fn ExplicitAndAlsoDeduced(T:! type, x: T) -> T*;

// TODO: This should presumably be accepted. We shouldn't deduce values for parameters with explicitly-specified values.
fn CallExplicitAndAlsoDeduced(n: i32) -> i32* {
  // CHECK:STDERR: fail_todo_explicit_vs_deduced.carbon:[[@LINE+7]]:10: error: inconsistent deductions for value of generic parameter `T`
  // CHECK:STDERR:   return ExplicitAndAlsoDeduced(A, {});
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_explicit_vs_deduced.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here
  // CHECK:STDERR: fn ExplicitAndAlsoDeduced(T:! type, x: T) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return ExplicitAndAlsoDeduced(A, {});
}

// --- deduce_implicit.carbon

library "[[@TEST_NAME]]";

fn ImplicitGenericParam[T:! type](x: T) -> T*;

fn CallImplicitGenericParam(n: i32) -> i32* {
  return ImplicitGenericParam(n);
}

// --- fail_todo_deduce_nested.carbon

library "[[@TEST_NAME]]";

fn TupleParam[T:! type](x: (T, i32));

fn CallTupleParam() {
  // CHECK:STDERR: fail_todo_deduce_nested.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T`
  // CHECK:STDERR:   TupleParam((1, 2));
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_nested.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here
  // CHECK:STDERR: fn TupleParam[T:! type](x: (T, i32));
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TupleParam((1, 2));
}

fn StructParam[T:! type](x: {.a: T, .b: i32});

fn CallStructParam() {
  // CHECK:STDERR: fail_todo_deduce_nested.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T`
  // CHECK:STDERR:   StructParam({.a = 1, .b = 2});
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_nested.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here
  // CHECK:STDERR: fn StructParam[T:! type](x: {.a: T, .b: i32});
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  StructParam({.a = 1, .b = 2});
}

// --- fail_deduce_incomplete.carbon

library "[[@TEST_NAME]]";

// TODO: It would be nice to diagnose this at its point of declaration, because
// U is not deducible.
fn ImplicitNotDeducible[T:! type, U:! type](x: T) -> U;

fn CallImplicitNotDeducible() {
  // CHECK:STDERR: fail_deduce_incomplete.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `U`
  // CHECK:STDERR:   ImplicitNotDeducible(42);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_incomplete.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here
  // CHECK:STDERR: fn ImplicitNotDeducible[T:! type, U:! type](x: T) -> U;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ImplicitNotDeducible(42);
}

// --- fail_deduce_inconsistent.carbon

library "[[@TEST_NAME]]";

fn ImplicitNotDeducible[T:! type](x: T, y: T) -> T;

fn CallImplicitNotDeducible() {
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE+6]]:3: error: inconsistent deductions for value of generic parameter `T`
  // CHECK:STDERR:   ImplicitNotDeducible(42, {.x = 12});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_inconsistent.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here
  // CHECK:STDERR: fn ImplicitNotDeducible[T:! type](x: T, y: T) -> T;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ImplicitNotDeducible(42, {.x = 12});
}

// CHECK:STDOUT: --- deduce_explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitGenericParam.type: type = fn_type @ExplicitGenericParam [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitGenericParam: %ExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallExplicitGenericParam.type: type = fn_type @CallExplicitGenericParam [template]
// CHECK:STDOUT:   %CallExplicitGenericParam: %CallExplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %.4: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [symbolic]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.type: type = fn_type @CallExplicitGenericParamWithGenericArg [template]
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg: %CallExplicitGenericParamWithGenericArg.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %.loc4_26: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_39: type = ptr_type %T [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:     %return: ref @ExplicitGenericParam.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParam.decl: %CallExplicitGenericParam.type = fn_decl @CallExplicitGenericParam [template = constants.%CallExplicitGenericParam] {} {
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_37.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_37.2: type = converted %int.make_type_32.loc6, %.loc6_37.1 [template = i32]
// CHECK:STDOUT:     %.loc6_37.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %return: ref %.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitGenericParamWithGenericArg.decl: %CallExplicitGenericParamWithGenericArg.type = fn_decl @CallExplicitGenericParamWithGenericArg [template = constants.%CallExplicitGenericParamWithGenericArg] {
// CHECK:STDOUT:     %T.patt.loc10: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %.loc10_44: type = param_pattern %T.patt.loc10, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10: type = name_ref T, %T.loc10 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc10_62: type = struct_type {.a: %T} [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:     %.loc10_63: type = ptr_type %.4 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:     %return: ref @CallExplicitGenericParamWithGenericArg.%.2 (%.5) = var <return slot>
// CHECK:STDOUT:     %param: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T 0, %param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ExplicitGenericParam(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = ptr_type @ExplicitGenericParam.%T.1 (%T) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc4_26: type) -> @ExplicitGenericParam.%.1 (%.1);
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
// CHECK:STDOUT:   %ExplicitGenericParam.call: init %.3 = call %ExplicitGenericParam.ref()
// CHECK:STDOUT:   %.loc7_35.1: %.3 = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_35.2: %.3 = converted %ExplicitGenericParam.call, %.loc7_35.1
// CHECK:STDOUT:   return %.loc7_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallExplicitGenericParamWithGenericArg(%T.loc10: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   %.1: type = struct_type {.a: @CallExplicitGenericParamWithGenericArg.%T.1 (%T)} [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: type = ptr_type @CallExplicitGenericParamWithGenericArg.%.1 (%.4) [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc10_44: type) -> @CallExplicitGenericParamWithGenericArg.%.2 (%.5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %ExplicitGenericParam.ref: %ExplicitGenericParam.type = name_ref ExplicitGenericParam, file.%ExplicitGenericParam.decl [template = constants.%ExplicitGenericParam]
// CHECK:STDOUT:     %T.ref.loc11: type = name_ref T, %T.loc10 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_37: type = struct_type {.a: %T} [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:     %ExplicitGenericParam.call: init @CallExplicitGenericParamWithGenericArg.%.2 (%.5) = call %ExplicitGenericParam.ref()
// CHECK:STDOUT:     %.loc11_39.1: @CallExplicitGenericParamWithGenericArg.%.2 (%.5) = value_of_initializer %ExplicitGenericParam.call
// CHECK:STDOUT:     %.loc11_39.2: @CallExplicitGenericParamWithGenericArg.%.2 (%.5) = converted %ExplicitGenericParam.call, %.loc11_39.1
// CHECK:STDOUT:     return %.loc11_39.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(i32) {
// CHECK:STDOUT:   %T.1 => i32
// CHECK:STDOUT:   %T.patt.1 => i32
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallExplicitGenericParamWithGenericArg(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitGenericParam(constants.%.4) {
// CHECK:STDOUT:   %T.1 => constants.%.4
// CHECK:STDOUT:   %T.patt.1 => constants.%.4
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_explicit_vs_deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.type: type = fn_type @ExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced: %ExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.type: type = fn_type @CallExplicitAndAlsoDeduced [template]
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced: %CallExplicitAndAlsoDeduced.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %.loc6_28: type = param_pattern %T.patt.loc6, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ExplicitAndAlsoDeduced.%T.1 (%T) = binding_pattern x
// CHECK:STDOUT:     %.loc6_38: @ExplicitAndAlsoDeduced.%T.1 (%T) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_40: type = name_ref T, %T.loc6 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc6_46: type = name_ref T, %T.loc6 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_47: type = ptr_type %T [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:     %return: ref @ExplicitAndAlsoDeduced.%.1 (%.3) = var <return slot>
// CHECK:STDOUT:     %param.loc6_27: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T 0, %param.loc6_27 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc6_37: @ExplicitAndAlsoDeduced.%T.1 (%T) = param runtime_param0
// CHECK:STDOUT:     %x: @ExplicitAndAlsoDeduced.%T.1 (%T) = bind_name x, %param.loc6_37
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallExplicitAndAlsoDeduced.decl: %CallExplicitAndAlsoDeduced.type = fn_decl @CallExplicitAndAlsoDeduced [template = constants.%CallExplicitAndAlsoDeduced] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %.loc9_32: i32 = param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc9_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_34.1: type = value_of_initializer %int.make_type_32.loc9_34 [template = i32]
// CHECK:STDOUT:     %.loc9_34.2: type = converted %int.make_type_32.loc9_34, %.loc9_34.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc9_42: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_45.1: type = value_of_initializer %int.make_type_32.loc9_42 [template = i32]
// CHECK:STDOUT:     %.loc9_45.2: type = converted %int.make_type_32.loc9_42, %.loc9_45.1 [template = i32]
// CHECK:STDOUT:     %.loc9_45.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %return: ref %.5 = var <return slot>
// CHECK:STDOUT:     %param: i32 = param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %param
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
// CHECK:STDOUT: generic fn @ExplicitAndAlsoDeduced(%T.loc6: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   %.1: type = ptr_type @ExplicitAndAlsoDeduced.%T.1 (%T) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc6_28: type, %.loc6_38: @ExplicitAndAlsoDeduced.%T.1 (%T)) -> @ExplicitAndAlsoDeduced.%.1 (%.3);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallExplicitAndAlsoDeduced(%.loc9_32: i32) -> %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ExplicitAndAlsoDeduced.ref: %ExplicitAndAlsoDeduced.type = name_ref ExplicitAndAlsoDeduced, file.%ExplicitAndAlsoDeduced.decl [template = constants.%ExplicitAndAlsoDeduced]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc17: %.1 = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ExplicitAndAlsoDeduced(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduce_implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %ImplicitGenericParam.type: type = fn_type @ImplicitGenericParam [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitGenericParam: %ImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %CallImplicitGenericParam.type: type = fn_type @CallImplicitGenericParam [template]
// CHECK:STDOUT:   %CallImplicitGenericParam: %CallImplicitGenericParam.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %.loc4_26: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitGenericParam.%T.1 (%T) = binding_pattern x
// CHECK:STDOUT:     %.loc4_36: @ImplicitGenericParam.%T.1 (%T) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_45: type = ptr_type %T [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:     %return: ref @ImplicitGenericParam.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:     %param.loc4_25: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param.loc4_25 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc4_35: @ImplicitGenericParam.%T.1 (%T) = param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitGenericParam.%T.1 (%T) = bind_name x, %param.loc4_35
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitGenericParam.decl: %CallImplicitGenericParam.type = fn_decl @CallImplicitGenericParam [template = constants.%CallImplicitGenericParam] {
// CHECK:STDOUT:     %n.patt: i32 = binding_pattern n
// CHECK:STDOUT:     %.loc6_30: i32 = param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_32.1: type = value_of_initializer %int.make_type_32.loc6_32 [template = i32]
// CHECK:STDOUT:     %.loc6_32.2: type = converted %int.make_type_32.loc6_32, %.loc6_32.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_32.loc6_40 [template = i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_32.loc6_40, %.loc6_43.1 [template = i32]
// CHECK:STDOUT:     %.loc6_43.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:     %return: ref %.3 = var <return slot>
// CHECK:STDOUT:     %param: i32 = param runtime_param0
// CHECK:STDOUT:     %n: i32 = bind_name n, %param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitGenericParam(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   %.1: type = ptr_type @ImplicitGenericParam.%T.1 (%T) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%.loc4_26: type](%.loc4_36: @ImplicitGenericParam.%T.1 (%T)) -> @ImplicitGenericParam.%.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitGenericParam(%.loc6_30: i32) -> %.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitGenericParam.ref: %ImplicitGenericParam.type = name_ref ImplicitGenericParam, file.%ImplicitGenericParam.decl [template = constants.%ImplicitGenericParam]
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %ImplicitGenericParam.call: init %.3 = call %ImplicitGenericParam.ref(%n.ref)
// CHECK:STDOUT:   %.loc7_33.1: %.3 = value_of_initializer %ImplicitGenericParam.call
// CHECK:STDOUT:   %.loc7_33.2: %.3 = converted %ImplicitGenericParam.call, %.loc7_33.1
// CHECK:STDOUT:   return %.loc7_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitGenericParam(i32) {
// CHECK:STDOUT:   %T.1 => i32
// CHECK:STDOUT:   %T.patt.1 => i32
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_deduce_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (%T, i32) [symbolic]
// CHECK:STDOUT:   %TupleParam.type: type = fn_type @TupleParam [template]
// CHECK:STDOUT:   %TupleParam: %TupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallTupleParam.type: type = fn_type @CallTupleParam [template]
// CHECK:STDOUT:   %CallTupleParam: %CallTupleParam.type = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %.7: type = struct_type {.a: %T, .b: i32} [symbolic]
// CHECK:STDOUT:   %StructParam.type: type = fn_type @StructParam [template]
// CHECK:STDOUT:   %StructParam: %StructParam.type = struct_value () [template]
// CHECK:STDOUT:   %CallStructParam.type: type = fn_type @CallStructParam [template]
// CHECK:STDOUT:   %CallStructParam: %CallStructParam.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TupleParam = %TupleParam.decl
// CHECK:STDOUT:     .CallTupleParam = %CallTupleParam.decl
// CHECK:STDOUT:     .StructParam = %StructParam.decl
// CHECK:STDOUT:     .CallStructParam = %CallStructParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleParam.decl: %TupleParam.type = fn_decl @TupleParam [template = constants.%TupleParam] {
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %.loc4_16: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:     %x.patt: @TupleParam.%.1 (%.3) = binding_pattern x
// CHECK:STDOUT:     %.loc4_26: @TupleParam.%.1 (%.3) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_35.1: %.2 = tuple_literal (%T.ref, %int.make_type_32)
// CHECK:STDOUT:     %.loc4_35.2: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_35.3: type = converted %int.make_type_32, %.loc4_35.2 [template = i32]
// CHECK:STDOUT:     %.loc4_35.4: type = converted %.loc4_35.1, constants.%.3 [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:     %param.loc4_15: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param.loc4_15 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc4_25: @TupleParam.%.1 (%.3) = param runtime_param0
// CHECK:STDOUT:     %x: @TupleParam.%.1 (%.3) = bind_name x, %param.loc4_25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallTupleParam.decl: %CallTupleParam.type = fn_decl @CallTupleParam [template = constants.%CallTupleParam] {} {}
// CHECK:STDOUT:   %StructParam.decl: %StructParam.type = fn_decl @StructParam [template = constants.%StructParam] {
// CHECK:STDOUT:     %T.patt.loc17: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %.loc17_17: type = param_pattern %T.patt.loc17, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:     %x.patt: @StructParam.%.1 (%.7) = binding_pattern x
// CHECK:STDOUT:     %.loc17_27: @StructParam.%.1 (%.7) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc17 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_41.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc17_41.2: type = converted %int.make_type_32, %.loc17_41.1 [template = i32]
// CHECK:STDOUT:     %.loc17_44: type = struct_type {.a: %T, .b: i32} [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:     %param.loc17_16: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc17: type = bind_symbolic_name T 0, %param.loc17_16 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc17_26: @StructParam.%.1 (%.7) = param runtime_param0
// CHECK:STDOUT:     %x: @StructParam.%.1 (%.7) = bind_name x, %param.loc17_26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallStructParam.decl: %CallStructParam.type = fn_decl @CallStructParam [template = constants.%CallStructParam] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TupleParam(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.1)]
// CHECK:STDOUT:   %.1: type = tuple_type (@TupleParam.%T.1 (%T), i32) [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%.loc4_16: type](%.loc4_26: @TupleParam.%.1 (%.3));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTupleParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TupleParam.ref: %TupleParam.type = name_ref TupleParam, file.%TupleParam.decl [template = constants.%TupleParam]
// CHECK:STDOUT:   %.loc14_15: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_18: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_19: %.6 = tuple_literal (%.loc14_15, %.loc14_18)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @StructParam(%T.loc17: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt.2)]
// CHECK:STDOUT:   %.1: type = struct_type {.a: @StructParam.%T.1 (%T), .b: i32} [symbolic = %.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%.loc17_17: type](%.loc17_27: @StructParam.%.1 (%.7));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallStructParam() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %StructParam.ref: %StructParam.type = name_ref StructParam, file.%StructParam.decl [template = constants.%StructParam]
// CHECK:STDOUT:   %.loc27_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc27_29: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc27_30: %.8 = struct_literal (%.loc27_21, %.loc27_29)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TupleParam(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @StructParam(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U 1 [symbolic]
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %.loc6_26: type = param_pattern %T.patt.loc6, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc6: type = symbolic_binding_pattern U 1 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:     %.loc6_36: type = param_pattern %U.patt.loc6, runtime_param<invalid> [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.1 (%T) = binding_pattern x
// CHECK:STDOUT:     %.loc6_46: @ImplicitNotDeducible.%T.1 (%T) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc6 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%U.1 (%U) = var <return slot>
// CHECK:STDOUT:     %param.loc6_25: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T 0, %param.loc6_25 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc6_35: type = param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc6: type = bind_symbolic_name U 1, %param.loc6_35 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:     %param.loc6_45: @ImplicitNotDeducible.%T.1 (%T) = param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.1 (%T) = bind_name x, %param.loc6_45
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [template = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc6: type, %U.loc6: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic = %U.1 (constants.%U)]
// CHECK:STDOUT:   %U.patt.1: type = symbolic_binding_pattern U 1 [symbolic = %U.patt.1 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%.loc6_26: type, %.loc6_36: type](%.loc6_46: @ImplicitNotDeducible.%T.1 (%T)) -> @ImplicitNotDeducible.%U.1 (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [template = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %.loc16: i32 = int_literal 42 [template = constants.%.2]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT:   %U.1 => constants.%U
// CHECK:STDOUT:   %U.patt.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_inconsistent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T 0 [symbolic]
// CHECK:STDOUT:   %ImplicitNotDeducible.type: type = fn_type @ImplicitNotDeducible [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %ImplicitNotDeducible: %ImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible.type: type = fn_type @CallImplicitNotDeducible [template]
// CHECK:STDOUT:   %CallImplicitNotDeducible: %CallImplicitNotDeducible.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.x: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt.loc4: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %.loc4_26: type = param_pattern %T.patt.loc4, runtime_param<invalid> [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @ImplicitNotDeducible.%T.1 (%T) = binding_pattern x
// CHECK:STDOUT:     %.loc4_36: @ImplicitNotDeducible.%T.1 (%T) = param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %y.patt: @ImplicitNotDeducible.%T.1 (%T) = binding_pattern y
// CHECK:STDOUT:     %.loc4_42: @ImplicitNotDeducible.%T.1 (%T) = param_pattern %y.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_38: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_44: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc4_50: type = name_ref T, %T.loc4 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %return: ref @ImplicitNotDeducible.%T.1 (%T) = var <return slot>
// CHECK:STDOUT:     %param.loc4_25: type = param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4: type = bind_symbolic_name T 0, %param.loc4_25 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %param.loc4_35: @ImplicitNotDeducible.%T.1 (%T) = param runtime_param0
// CHECK:STDOUT:     %x: @ImplicitNotDeducible.%T.1 (%T) = bind_name x, %param.loc4_35
// CHECK:STDOUT:     %param.loc4_41: @ImplicitNotDeducible.%T.1 (%T) = param runtime_param1
// CHECK:STDOUT:     %y: @ImplicitNotDeducible.%T.1 (%T) = bind_name y, %param.loc4_41
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallImplicitNotDeducible.decl: %CallImplicitNotDeducible.type = fn_decl @CallImplicitNotDeducible [template = constants.%CallImplicitNotDeducible] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitNotDeducible(%T.loc4: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.1: type = symbolic_binding_pattern T 0 [symbolic = %T.patt.1 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%.loc4_26: type](%.loc4_36: @ImplicitNotDeducible.%T.1 (%T), %.loc4_42: @ImplicitNotDeducible.%T.1 (%T)) -> @ImplicitNotDeducible.%T.1 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImplicitNotDeducible() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ImplicitNotDeducible.ref: %ImplicitNotDeducible.type = name_ref ImplicitNotDeducible, file.%ImplicitNotDeducible.decl [template = constants.%ImplicitNotDeducible]
// CHECK:STDOUT:   %.loc13_24: i32 = int_literal 42 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_34: i32 = int_literal 12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_36: %.4 = struct_literal (%.loc13_34)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitNotDeducible(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %T.patt.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
