// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/fail_paren_match_regression.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/fail_paren_match_regression.carbon

// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+8]]:5: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: var = (foo {})
// CHECK:STDERR:     ^
// CHECK:STDERR:
// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+4]]:12: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
// CHECK:STDERR: var = (foo {})
// CHECK:STDERR:            ^
// CHECK:STDERR:
var = (foo {})

// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+18]]:21: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_paren_match_regression.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: '=', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: '=', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ')', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
