// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/var_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/var_tuple.carbon

fn F() {
  var (s: String, i: 32) = ("hello", 0);
}

// CHECK:STDOUT: - filename: var_tuple.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 's'},
// CHECK:STDOUT:               {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'i'},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '32'},
// CHECK:STDOUT:             {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
