// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

virtual class B

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1: ERROR: `virtual` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: impl class
// CHECK:STDERR: ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:6: ERROR: `class` introducer should be followed by a name.
// CHECK:STDERR: impl class
// CHECK:STDERR:      ^~~~~
impl class

// CHECK:STDOUT: - filename: fail_modifiers.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: 'impl', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: 'class', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
