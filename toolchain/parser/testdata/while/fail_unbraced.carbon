// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'WhileConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:     {kind: 'WhileCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: 'break', has_error: yes},
// CHECK:STDOUT:         {kind: 'BreakStatementStart', text: 'break'},
// CHECK:STDOUT:       {kind: 'BreakStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: 'break', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:   {kind: 'WhileStatement', text: 'while', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  while (a)
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/while/fail_unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    break;
}
