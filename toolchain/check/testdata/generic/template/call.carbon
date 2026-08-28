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

// --- fail_dependent_call_nonconst_arg.carbon
library "[[@TEST_NAME]]";

class X {}
class Y {}
fn F[template G: type](H: G, x: X) {
  H(x);
}
fn TakeX(unused x: X) {}
fn TakeY(unused y: Y) {}

// TODO: should work
// CHECK:STDERR: fail_dependent_call_nonconst_arg.carbon:[[@LINE+7]]:23: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fn GoodCall(x: X) { F(TakeX, x); }
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_dependent_call_nonconst_arg.carbon:[[@LINE-10]]:24: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn F[template G: type](H: G, x: X) {
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
fn GoodCall(x: X) { F(TakeX, x); }

// TODO: should fail, but not because of UseOfNonExprAsValue
// CHECK:STDERR: fail_dependent_call_nonconst_arg.carbon:[[@LINE+7]]:22: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fn BadCall(x: X) { F(TakeY, x); }
// CHECK:STDERR:                      ^~~~~
// CHECK:STDERR: fail_dependent_call_nonconst_arg.carbon:[[@LINE-20]]:24: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: fn F[template G: type](H: G, x: X) {
// CHECK:STDERR:                        ^~~~
// CHECK:STDERR:
fn BadCall(x: X) { F(TakeY, x); }
