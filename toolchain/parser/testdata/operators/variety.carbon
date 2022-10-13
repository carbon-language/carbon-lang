// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

fn F() {
  // CHECK:{{.*}}/toolchain/parser/testdata/operators/variety.carbon:[[@LINE+34]]:29: Parentheses are required to disambiguate operator precedence.
  // CHECK:{{.*}}/toolchain/parser/testdata/operators/variety.carbon:[[@LINE+33]]:34: Parentheses are required to disambiguate operator precedence.
  // CHECK:{{.*}}/toolchain/parser/testdata/operators/variety.carbon:[[@LINE+32]]:38: Parentheses are required to disambiguate operator precedence.
  // CHECK:{{.*}}/toolchain/parser/testdata/operators/variety.carbon:[[@LINE+31]]:40: Parentheses are required to disambiguate operator precedence.
  // CHECK:[
  // CHECK:{node_index: 26, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 27, children: [
  // CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'F'},
  // CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
  // CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
  // CHECK:  {node_index: 25, kind: 'CodeBlock', text: '{', subtree_size: 23, children: [
  // CHECK:    {node_index: 23, kind: 'ExpressionStatement', text: ';', subtree_size: 21, children: [
  // CHECK:      {node_index: 22, kind: 'InfixOperator', text: '=', subtree_size: 20, children: [
  // CHECK:        {node_index: 3, kind: 'NameReference', text: 'n'},
  // CHECK:        {node_index: 21, kind: 'InfixOperator', text: '=', subtree_size: 18, children: [
  // CHECK:          {node_index: 10, kind: 'InfixOperator', text: '+', subtree_size: 7, children: [
  // CHECK:            {node_index: 6, kind: 'InfixOperator', text: '*', subtree_size: 3, children: [
  // CHECK:              {node_index: 4, kind: 'NameReference', text: 'a'},
  // CHECK:              {node_index: 5, kind: 'NameReference', text: 'b'}]},
  // CHECK:            {node_index: 9, kind: 'InfixOperator', text: '*', subtree_size: 3, children: [
  // CHECK:              {node_index: 7, kind: 'NameReference', text: 'c'},
  // CHECK:              {node_index: 8, kind: 'NameReference', text: 'd'}]}]},
  // CHECK:          {node_index: 20, kind: 'InfixOperator', text: '-', has_error: yes, subtree_size: 10, children: [
  // CHECK:            {node_index: 17, kind: 'InfixOperator', text: '&', has_error: yes, subtree_size: 7, children: [
  // CHECK:              {node_index: 15, kind: 'InfixOperator', text: '<<', has_error: yes, subtree_size: 5, children: [
  // CHECK:                {node_index: 13, kind: 'InfixOperator', text: '*', subtree_size: 3, children: [
  // CHECK:                  {node_index: 11, kind: 'NameReference', text: 'd'},
  // CHECK:                  {node_index: 12, kind: 'NameReference', text: 'd'}]},
  // CHECK:                {node_index: 14, kind: 'NameReference', text: 'e'}]},
  // CHECK:              {node_index: 16, kind: 'NameReference', text: 'f'}]},
  // CHECK:            {node_index: 19, kind: 'PrefixOperator', text: 'not', subtree_size: 2, children: [
  // CHECK:              {node_index: 18, kind: 'NameReference', text: 'g'}]}]}]}]}]},
  // CHECK:    {node_index: 24, kind: 'CodeBlockEnd', text: '}'}]}]},
  // CHECK:{node_index: 27, kind: 'FileEnd', text: ''},
  // CHECK:]
  n = a * b + c * d = d * d << e & f - not g;
}
