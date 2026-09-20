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

// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE+6]]:23-27: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE-6]]:24-27: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE+3]]:21: error: unable to monomorphize specific `F(<type of TakeX>)` [ResolvingSpecificHere]
// CHECK:STDERR: fail_todo_dependent_call_nonconst_arg.carbon:[[@LINE-8]]:3-6: note: value of type `<type of TakeX>` is not callable [CallToNonCallable]
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
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE+6]]:22-26: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE-7]]:24-27: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE+3]]:20: error: unable to monomorphize specific `F(<type of TakeY>)` [ResolvingSpecificHere]
// CHECK:STDERR: fail_dependent_call_nonconst_arg_wrong_type.carbon:[[@LINE-9]]:3-6: note: value of type `<type of TakeY>` is not callable [CallToNonCallable]
// CHECK:STDERR:
fn BadCall(x: X) { F(TakeY, x); }
