// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_infix_uneven_space_after.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_infix_uneven_space_after.carbon

// TODO: We could figure out that this first Failed example is infix
// with one-token lookahead.
// CHECK:STDERR: fail_infix_uneven_space_after.carbon:[[@LINE+4]]:16: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: var n: i8 = n* n;
// CHECK:STDERR:                ^
// CHECK:STDERR:
var n: i8 = n* n;

// CHECK:STDOUT: - filename: fail_infix_uneven_space_after.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i8'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:       {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
