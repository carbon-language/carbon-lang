// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/final_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/final_fn.carbon

interface Order {
  fn Less[self: Self](right: Self) -> bool;
  final fn Greater[self: Self](right: Self) -> bool {
    return right.Less(self);
  }
}

// CHECK:STDOUT: - filename: final_fn.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Order'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Less'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'right'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'Greater'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'right'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:           {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 16},
// CHECK:STDOUT:           {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'right'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'Less'},
// CHECK:STDOUT:               {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'CallExprStart', text: '(', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'SelfValueNameExpr', text: 'self'},
// CHECK:STDOUT:           {kind: 'CallExpr', text: ')', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 44},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
