// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration/fail_with_identifier_as_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration/fail_with_identifier_as_param.carbon

// CHECK:STDERR: fail_with_identifier_as_param.carbon:[[@LINE+4]]:11: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn foo(bar);
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn foo(bar);

// CHECK:STDOUT: - filename: fail_with_identifier_as_param.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'bar'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'bar', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
