// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_self_param_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_self_param_syntax.carbon

interface Foo {
  // CHECK:STDERR: fail_self_param_syntax.carbon:[[@LINE+4]]:13: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn Sub[me Self](b: Self) -> Self;
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  fn Sub[me Self](b: Self) -> Self;

  // CHECK:STDERR: fail_self_param_syntax.carbon:[[@LINE+4]]:10: error: expected name in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn Mul[Self](b: Self) -> Self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  fn Mul[Self](b: Self) -> Self;
}

// CHECK:STDOUT: - filename: fail_self_param_syntax.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Sub'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'me'},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: 'me', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Mul'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'Self', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: 'Self', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
