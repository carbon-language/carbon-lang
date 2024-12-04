// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+12]]:11: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:           ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+8]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_: F(true and true);

// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+11]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+7]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_or_not_in_function.carbon:[[@LINE+3]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
var or_: F(true or true);

// CHECK:STDOUT: --- fail_and_or_not_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc42_17: bool = block_arg <unexpected instblockref block23> [template = constants.%true]
// CHECK:STDOUT:   %F.call: init type = call <unexpected>.inst+73.loc42_10(%.loc42_17)
// CHECK:STDOUT:   %.loc42_24.1: type = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc42_24.2: type = converted %F.call, %.loc42_24.1
// CHECK:STDOUT:   %or_.var: ref <error> = var or_
// CHECK:STDOUT:   %or_: ref <error> = bind_name or_, %or_.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param_patt: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst+13.loc12_6
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_20.2: type = converted %int.make_type_signed, %.loc13_20.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%.loc13_20.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [template = f64]
// CHECK:STDOUT:   %.loc13_24.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc13_24.2: type = converted %float.make_type, %.loc13_24.1 [template = f64]
// CHECK:STDOUT:   br !if.expr.result(%.loc13_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc13_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc13_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
