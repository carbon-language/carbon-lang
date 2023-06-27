// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:       {kind: 'Name', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:       {kind: 'Name', text: 'fn', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:       {kind: 'Name', text: '42', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.;
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.fn;
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.42;
}
