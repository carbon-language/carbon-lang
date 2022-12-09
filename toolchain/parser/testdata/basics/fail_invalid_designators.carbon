// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: 'fn', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: '42', has_error: yes},
// CHECK:STDOUT:     {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.;
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.fn;
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_invalid_designators.carbon:[[@LINE+1]]:5: Expected identifier after `.`.
  a.42;
}
