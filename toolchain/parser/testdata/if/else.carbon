// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:           {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:           {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'e'},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 24},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'f'},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:       {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'g'},
// CHECK:STDOUT:         {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'h'},
// CHECK:STDOUT:         {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'IfStatement', text: 'if', subtree_size: 13},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 22},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 52},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  if (a) {
    if (b) {
      c;
    } else {
      d;
    }
  } else {
    e;
  }
  if (x) { f; }
  else if (x) { g; }
  else { h; }
}
