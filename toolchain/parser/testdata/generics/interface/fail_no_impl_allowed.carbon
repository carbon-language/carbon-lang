// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'InterfaceBodyStart', text: '{'},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'SelfDeducedParameter', text: 'self'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 15},
// CHECK:STDOUT:   {kind: 'InterfaceBody', text: '}', subtree_size: 17},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', subtree_size: 19},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_no_impl_allowed.carbon:[[@LINE+1]]:39: Method implementations are not allowed in interfaces.
  fn Add[self: Self](b: Self) -> Self {
    print("You can't do that.");
  }
}
