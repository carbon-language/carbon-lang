// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/with_length.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/with_length.carbon

var x: [i32; 10];

// CHECK:STDOUT: - filename: with_length.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '10'},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
