// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/fail_todo_template_access_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/fail_todo_template_access_assoc_const.carbon

library "[[@TEST_NAME]]";

// TODO: This should check. The `T.I1` access should be deferred until we know
// the value of the `template T`.
//
// CHECK:STDERR: fail_todo_template_access_assoc_const.carbon:[[@LINE+4]]:36: error: cannot evaluate type expression [TypeExprEvaluationFailure]
// CHECK:STDERR: interface I(template T:! type, N:! T.I1) {
// CHECK:STDERR:                                    ^~~~
// CHECK:STDERR:
interface I(template T:! type, N:! T.I1) {
    let I1:! type;
}
