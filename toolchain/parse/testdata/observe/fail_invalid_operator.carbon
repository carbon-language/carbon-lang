// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/fail_invalid_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/fail_invalid_operator.carbon

// CHECK:STDERR: fail_invalid_operator.carbon:[[@LINE+4]]:11: error: observe should use `==` or `impls` operator [ExpectedObserveOperator]
// CHECK:STDERR: observe T = U impls I;
// CHECK:STDERR:           ^
// CHECK:STDERR:
observe T = U impls I;

// CHECK:STDOUT: - filename: fail_invalid_operator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ObserveIntroducer', text: 'observe'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:     {kind: 'ObserveDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
