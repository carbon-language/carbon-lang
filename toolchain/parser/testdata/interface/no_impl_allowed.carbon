// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 18, kind: 'InterfaceDeclaration', text: 'interface', subtree_size: 19, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 17, kind: 'InterfaceBody', text: '{', subtree_size: 17, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {node_index: 2, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 7, kind: 'MeParam', text: '[', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 5, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 3, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 4, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 6, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 12, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 10, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 8, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 9, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 11, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 14, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 13, kind: 'Self', text: 'Self'}]}]},
// CHECK:STDOUT:     {node_index: 16, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 19, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/interface/no_impl_allowed.carbon:[[@LINE+1]]:37: Method implementations are not allowed in interfaces.
  fn Add[me: Self](b: Self) -> Self {
    print("You can't do that.");
  }
}
