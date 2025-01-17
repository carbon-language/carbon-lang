// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_postfix_space_before_comma.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_postfix_space_before_comma.carbon

// CHECK:STDERR: fail_postfix_space_before_comma.carbon:[[@LINE+4]]:18: error: whitespace is not allowed before this unary operator [UnaryOperatorHasWhitespace]
// CHECK:STDERR: var n: i8 = F(i8 *, 0);
// CHECK:STDERR:                  ^
// CHECK:STDERR:
var n: i8 = F(i8 *, 0);

// CHECK:STDOUT: - filename: fail_postfix_space_before_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i8'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'F'},
// CHECK:STDOUT:         {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i8'},
// CHECK:STDOUT:         {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'CallExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
