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
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:                 {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:               {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 16},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 20},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  if (a)
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    if (b)
      // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:7: Expected braced code block.
      if (c)
        // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:9: Expected braced code block.
        d;
}
