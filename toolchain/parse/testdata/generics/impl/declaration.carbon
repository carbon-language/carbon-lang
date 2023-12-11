// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

impl bool as Interface;

// CHECK:STDOUT: - filename: declaration.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:       {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
