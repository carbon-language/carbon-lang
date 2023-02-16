// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ForHeaderStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'y'},
// CHECK:STDOUT:     {kind: 'ForHeader', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'Print'},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ForStatement', text: 'for', subtree_size: 11},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/for/fail_missing_var.carbon:[[@LINE+1]]:8: Expected `var` declaration.
  for (x: i32 in y) {
    Print(x);
  }
}
