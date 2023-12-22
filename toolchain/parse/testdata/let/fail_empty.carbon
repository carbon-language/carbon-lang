// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_empty.carbon:[[@LINE+3]]:4: ERROR: Expected binding pattern.
// CHECK:STDERR: let;
// CHECK:STDERR:    ^
let;

// CHECK:STDOUT: - filename: fail_empty.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
