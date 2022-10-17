// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 44, kind: 'InterfaceDeclaration', text: 'interface', subtree_size: 45, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 43, kind: 'InterfaceBody', text: '{', subtree_size: 43, children: [
//
// CHECK:STDOUT:     {node_index: 10, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 10, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 6, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 4, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 2, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 3, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 5, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 8, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 7, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 9, kind: 'DeclarationEnd', text: ';'}]},
//
// CHECK:STDOUT:     {node_index: 21, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 11, children: [
// CHECK:STDOUT:       {node_index: 11, kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {node_index: 12, kind: 'Me', text: 'me'},
// CHECK:STDOUT:       {node_index: 17, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 15, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 13, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 14, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 16, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 19, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 18, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 20, kind: 'DeclarationEnd', text: ';'}]},
//
// CHECK:STDOUT:     {node_index: 31, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 10, children: [
// CHECK:STDOUT:       {node_index: 22, kind: 'DeclaredName', text: 'Mul'},
// CHECK:STDOUT:       {node_index: 27, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 25, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 23, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 24, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 26, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 29, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 28, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 30, kind: 'DeclarationEnd', text: ';'}]},
//
// CHECK:STDOUT:     {node_index: 41, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 10, children: [
// CHECK:STDOUT:       {node_index: 32, kind: 'DeclaredName', text: 'Rem'},
// CHECK:STDOUT:       {node_index: 37, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 35, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 33, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 34, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 36, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 39, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 38, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 40, kind: 'DeclarationEnd', text: ';'}]},
//
// CHECK:STDOUT:     {node_index: 42, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 45, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface foo {
  fn Add(b: Self) -> Self;

  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/interface/me_param_required.carbon:[[@LINE+1]]:13: Associated method `Sub` must have a `me` parameter of the form: `[me: Self] or [addr me: Self*]`.
  fn Sub[me Self](b: Self) -> Self;

  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/interface/me_param_required.carbon:[[@LINE+1]]:10: Associated method `Mul` must have a `me` parameter of the form: `[me: Self] or [addr me: Self*]`.
  fn Mul[Self](b: Self) -> Self;

  // TODO The recovery token is currently inserted after the `;`:
  // ```
  // fn Div[me: Self(b: Self) -> Self;]
  // ```
  // Fix and uncomment this to test error handling.
  // fn Div[me: Self(b: Self) -> Self;

  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/interface/me_param_required.carbon:[[@LINE+1]]:10: Associated method `Rem` must have a `me` parameter of the form: `[me: Self] or [addr me: Self*]`.
  fn Rem[](b: Self) -> Self;
}
