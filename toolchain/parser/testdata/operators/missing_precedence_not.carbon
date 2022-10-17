// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 14, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 15, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 13, kind: 'CodeBlock', text: '{', subtree_size: 11, children: [
// CHECK:STDOUT:     {node_index: 11, kind: 'ExpressionStatement', text: ';', subtree_size: 9, children: [
// CHECK:STDOUT:       {node_index: 10, kind: 'InfixOperator', text: 'and', subtree_size: 8, children: [
// CHECK:STDOUT:         {node_index: 7, kind: 'InfixOperator', text: 'and', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 4, kind: 'PrefixOperator', text: 'not', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 3, kind: 'NameReference', text: 'a'}]},
// CHECK:STDOUT:           {node_index: 6, kind: 'PrefixOperator', text: 'not', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 5, kind: 'NameReference', text: 'b'}]}]},
// CHECK:STDOUT:         {node_index: 9, kind: 'PrefixOperator', text: 'not', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 8, kind: 'NameReference', text: 'c'}]}]}]},
// CHECK:STDOUT:     {node_index: 12, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 15, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+3]]:3: Whitespace is required before this unary operator.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+2]]:13: Whitespace is required before this unary operator.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+1]]:23: Whitespace is required before this unary operator.
  not a and not b and not c;
}
