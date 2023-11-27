// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

impl fn F();
abstract impl fn G();
impl abstract fn H();
private impl default fn I();

// CHECK:STDOUT: - filename: impl_fn.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'impl'},
// CHECK:STDOUT:       {kind: 'Name', text: 'F'},
// CHECK:STDOUT:         {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'impl'},
// CHECK:STDOUT:       {kind: 'Name', text: 'G'},
// CHECK:STDOUT:         {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'impl'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'Name', text: 'H'},
// CHECK:STDOUT:         {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'AccessModifierKeyword', text: 'private'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'impl'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'default'},
// CHECK:STDOUT:       {kind: 'Name', text: 'I'},
// CHECK:STDOUT:         {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
