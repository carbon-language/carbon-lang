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
// CHECK:STDOUT:                   {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:                   {kind: 'DesignatedName', text: 'b'},
// CHECK:STDOUT:                 {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'DesignatedName', text: 'f'},
// CHECK:STDOUT:               {kind: 'DesignatorExpression', text: '.', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'CallExpressionStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:               {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:               {kind: 'DesignatedName', text: 'd'},
// CHECK:STDOUT:             {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:               {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'NameReference', text: 'e'},
// CHECK:STDOUT:             {kind: 'ParenExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CallExpression', text: ')', subtree_size: 14},
// CHECK:STDOUT:           {kind: 'DesignatedName', text: 'g'},
// CHECK:STDOUT:         {kind: 'DesignatorExpression', text: '.', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'CallExpressionStart', text: '(', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'CallExpression', text: ')', subtree_size: 18},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 19},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  a.b.f(c.d, (e)).g();
}
