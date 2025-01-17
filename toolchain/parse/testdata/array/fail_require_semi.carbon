// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/fail_require_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/fail_require_semi.carbon

// CHECK:STDERR: fail_require_semi.carbon:[[@LINE+4]]:12: error: expected `;` in array type [ExpectedArraySemi]
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
// CHECK:STDERR:
var x: [i32];

// CHECK:STDOUT: - filename: fail_require_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
