// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration/fail_missing_implicit_close.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration/fail_missing_implicit_close.carbon

// Fix and uncomment this to test error handling.
// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+8]]:7: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+4]]:7: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
// CHECK:STDERR:
fn Div[();

// CHECK:STDOUT: - filename: fail_missing_implicit_close.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Div'},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
