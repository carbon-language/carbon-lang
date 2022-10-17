// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 24, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 25, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 23, kind: 'CodeBlock', text: '{', subtree_size: 21, children: [
// CHECK:STDOUT:     {node_index: 21, kind: 'ExpressionStatement', text: ';', subtree_size: 19, children: [
// CHECK:STDOUT:       {node_index: 20, kind: 'CallExpression', text: '(', subtree_size: 18, children: [
// CHECK:STDOUT:         {node_index: 18, kind: 'DesignatorExpression', text: '.', subtree_size: 16, children: [
// CHECK:STDOUT:           {node_index: 16, kind: 'CallExpression', text: '(', subtree_size: 14, children: [
// CHECK:STDOUT:             {node_index: 7, kind: 'DesignatorExpression', text: '.', subtree_size: 5, children: [
// CHECK:STDOUT:               {node_index: 5, kind: 'DesignatorExpression', text: '.', subtree_size: 3, children: [
// CHECK:STDOUT:                 {node_index: 3, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:                 {node_index: 4, kind: 'DesignatedName', text: 'b'}]},
// CHECK:STDOUT:               {node_index: 6, kind: 'DesignatedName', text: 'f'}]},
// CHECK:STDOUT:             {node_index: 10, kind: 'DesignatorExpression', text: '.', subtree_size: 3, children: [
// CHECK:STDOUT:               {node_index: 8, kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:               {node_index: 9, kind: 'DesignatedName', text: 'd'}]},
// CHECK:STDOUT:             {node_index: 11, kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:             {node_index: 14, kind: 'ParenExpression', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:               {node_index: 12, kind: 'NameReference', text: 'e'},
// CHECK:STDOUT:               {node_index: 13, kind: 'ParenExpressionEnd', text: ')'}]},
// CHECK:STDOUT:             {node_index: 15, kind: 'CallExpressionEnd', text: ')'}]},
// CHECK:STDOUT:           {node_index: 17, kind: 'DesignatedName', text: 'g'}]},
// CHECK:STDOUT:         {node_index: 19, kind: 'CallExpressionEnd', text: ')'}]}]},
// CHECK:STDOUT:     {node_index: 22, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 25, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  a.b.f(c.d, (e)).g();
}
