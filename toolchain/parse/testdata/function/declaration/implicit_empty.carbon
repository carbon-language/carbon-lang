// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn foo[]();

// CHECK:STDOUT: - filename: implicit_empty.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'Name', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
