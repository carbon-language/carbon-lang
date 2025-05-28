// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/int_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/int_float.carbon

// --- int.carbon

library "[[@TEST_NAME]]";

fn F[N:! Core.IntLiteral()](n: Core.Int(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: i64) -> Core.IntLiteral() {
  return F(a);
}

// --- fail_todo_float.carbon

library "[[@TEST_NAME]]";

// TODO: This should pass once we accept symbolic widths for Core.Float.
// CHECK:STDERR: fail_todo_float.carbon:[[@LINE+4]]:32: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: fn F[N:! Core.IntLiteral()](n: Core.Float(N)) -> Core.IntLiteral() {
// CHECK:STDERR:                                ^~~~~~~~~~~~~
// CHECK:STDERR:
fn F[N:! Core.IntLiteral()](n: Core.Float(N)) -> Core.IntLiteral() {
  return N;
}

fn G(a: f64) -> Core.IntLiteral() {
  return F(a);
}
