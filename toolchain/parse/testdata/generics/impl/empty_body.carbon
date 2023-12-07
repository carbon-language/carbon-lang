// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

impl String as Interface {
}

// CHECK:STDOUT: - filename: empty_body.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
