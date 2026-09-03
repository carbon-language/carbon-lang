// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/call.carbon

// --- fail_todo_dependent_call_nonconst_arg.carbon
library "[[@TEST_NAME]]";

class X {}
class Y {}
fn F[template G: type](H: G, x: X) {
  H(x);
}
fn TakeX(unused x: X) {}

// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE+14]]:23: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fn GoodCall(x: X) { F(TakeX, x); }
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE-8]]:24: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn F[template G: type](H: G, x: X) {
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE+7]]:21: error: unable to monomorphize specific `F(<type of TakeX>)` [ResolvingSpecificHere]
// CHECK:STDERR: fn GoodCall(x: X) { F(TakeX, x); }
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE-14]]:3: note: value of type `<type of TakeX>` is not callable [CallToNonCallable]
// CHECK:STDERR:   H(x);
// CHECK:STDERR:   ^~~~
// CHECK:STDERR:
fn GoodCall(x: X) { F(TakeX, x); }

// --- fail_dependent_call_nonconst_arg_wrong_type.carbon
library "[[@TEST_NAME]]";

class X {}
class Y {}
fn F[template G: type](H: G, x: X) {
  H(x);
}
fn TakeY(unused y: Y) {}

// TODO: should fail, but not because of UseOfNonExprAsValue
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE+14]]:22: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fn BadCall(x: X) { F(TakeY, x); }
// CHECK:STDERR:                      ^~~~~
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE-9]]:24: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn F[template G: type](H: G, x: X) {
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE+7]]:20: error: unable to monomorphize specific `F(<type of TakeY>)` [ResolvingSpecificHere]
// CHECK:STDERR: fn BadCall(x: X) { F(TakeY, x); }
// CHECK:STDERR:                    ^
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE-15]]:3: note: value of type `<type of TakeY>` is not callable [CallToNonCallable]
// CHECK:STDERR:   H(x);
// CHECK:STDERR:   ^~~~
// CHECK:STDERR:
fn BadCall(x: X) { F(TakeY, x); }
