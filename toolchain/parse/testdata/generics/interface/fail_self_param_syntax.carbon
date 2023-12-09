// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Foo {
  // CHECK:STDERR: fail_self_param_syntax.carbon:[[@LINE+3]]:13: ERROR: Expected binding pattern.
  // CHECK:STDERR:   fn Sub[me Self](b: Self) -> Self;
  // CHECK:STDERR:             ^~~~
  fn Sub[me Self](b: Self) -> Self;

  // CHECK:STDERR: fail_self_param_syntax.carbon:[[@LINE+3]]:10: ERROR: Expected binding pattern.
  // CHECK:STDERR:   fn Mul[Self](b: Self) -> Self;
  // CHECK:STDERR:          ^~~~
  fn Mul[Self](b: Self) -> Self;
}

// CHECK:STDOUT: - filename: fail_self_param_syntax.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Sub'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'me'},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: 'me', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Mul'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'Self', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: 'Self', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
