// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 74, kind: 'InterfaceDeclaration', text: 'interface', subtree_size: 75, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 73, kind: 'InterfaceBody', text: '{', subtree_size: 73, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 6, kind: 'MeParam', text: '[', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 4, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 2, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 3, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 5, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 11, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 9, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 7, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 8, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 10, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 13, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 12, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 14, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 30, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 16, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 21, kind: 'MeParam', text: '[', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 19, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 17, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 18, kind: 'NameReference', text: 'foo'}]},
// CHECK:STDOUT:         {node_index: 20, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 26, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 24, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 22, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 23, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 25, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 28, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 27, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 29, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 47, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 17, children: [
// CHECK:STDOUT:       {node_index: 31, kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {node_index: 38, kind: 'MeParam', text: '[', subtree_size: 7, children: [
// CHECK:STDOUT:         {node_index: 36, kind: 'PatternBinding', text: ':', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 32, kind: 'Addr', text: 'addr'},
// CHECK:STDOUT:           {node_index: 33, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 35, kind: 'PostfixOperator', text: '*', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 34, kind: 'Self', text: 'Self'}]}]},
// CHECK:STDOUT:         {node_index: 37, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 43, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 41, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 39, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 40, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 42, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 45, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 44, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 46, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 64, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 17, children: [
// CHECK:STDOUT:       {node_index: 48, kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {node_index: 55, kind: 'MeParam', text: '[', subtree_size: 7, children: [
// CHECK:STDOUT:         {node_index: 53, kind: 'PatternBinding', text: ':', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 49, kind: 'Addr', text: 'addr'},
// CHECK:STDOUT:           {node_index: 50, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 52, kind: 'PostfixOperator', text: '*', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 51, kind: 'NameReference', text: 'foo'}]}]},
// CHECK:STDOUT:         {node_index: 54, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 60, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 58, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 56, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 57, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 59, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 62, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 61, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 63, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 71, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 65, kind: 'DeclaredName', text: 'FooFactory'},
// CHECK:STDOUT:       {node_index: 67, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 66, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 69, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 68, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 70, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 72, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 75, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface foo {
  fn Add[me: Self](b: Self) -> Self;
  fn Add[me: foo](b: Self) -> Self;

  fn Sub[addr me: Self*](b: Self) -> Self;
  fn Sub[addr me: foo*](b: Self) -> Self;

  fn FooFactory() -> Self;
}
