// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_not_in_function.carbon

// --- fail_and.carbon

library "[[@TEST_NAME]]";

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// TODO: Short-circuit operators should be permitted outside functions.
// CHECK:STDERR: fail_and.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_: F(true and true);
// CHECK:STDERR:             ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_: F(true and true);

// --- fail_and_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_and_val.carbon:[[@LINE+8]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_and_val.carbon:[[@LINE+4]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var and_val: bool = true and true;
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var and_val: bool = true and true;

// --- fail_or.carbon

library "[[@TEST_NAME]]";

// TODO: Make this a compile-time function.
fn F(b: bool) -> type {
  return if b then i32 else f64;
}

// CHECK:STDERR: fail_or.carbon:[[@LINE+4]]:12: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_: F(true or true);
// CHECK:STDERR:            ^~~~~~~~~~~~
// CHECK:STDERR:
var or_: F(true or true);

// --- fail_or_val.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_or_val.carbon:[[@LINE+8]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_or_val.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var or_val: bool = true or true;
// CHECK:STDERR:                    ^~~~~~~~~~~~
// CHECK:STDERR:
var or_val: bool = true or true;

// CHECK:STDOUT: --- fail_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst28.loc5_6
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:   %.loc6_24.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:   %.loc6_24.2: type = converted %float.make_type, %.loc6_24.1 [concrete = f64]
// CHECK:STDOUT:   br !if.expr.result(%.loc6_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc6_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_and_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: bool) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, <unexpected>.inst28.loc5_6
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:   %.loc6_24.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:   %.loc6_24.2: type = converted %float.make_type, %.loc6_24.1 [concrete = f64]
// CHECK:STDOUT:   br !if.expr.result(%.loc6_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc6_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc6_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_or_val.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
