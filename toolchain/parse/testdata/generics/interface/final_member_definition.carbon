// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/final_member_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/final_member_definition.carbon

interface Foo {
  final fn Add[self: Self](b: Self) -> Self {
    print("Allowed.");
  }
}

// CHECK:STDOUT: - filename: final_member_definition.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'Add'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 16},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'print'},
// CHECK:STDOUT:             {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'StringLiteral', text: '"Allowed."'},
// CHECK:STDOUT:           {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
