// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 15, kind: 'FunctionDefinition', text: '}', subtree_size: 16, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 7, kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:     {node_index: 6, kind: 'DesignatorExpression', text: '.', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'NameReference', text: 'a'}]}]},
// CHECK:STDOUT:   {node_index: 11, kind: 'ExpressionStatement', text: ';', subtree_size: 4, children: [
// CHECK:STDOUT:     {node_index: 10, kind: 'DesignatorExpression', text: '.', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {node_index: 9, kind: 'DesignatedName', text: 'fn', has_error: yes}]}]},
// CHECK:STDOUT:   {node_index: 14, kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:     {node_index: 13, kind: 'DesignatorExpression', text: '.', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 12, kind: 'NameReference', text: 'a'}]}]}]},
// CHECK:STDOUT: {node_index: 16, kind: 'FileEnd', text: ''},
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
