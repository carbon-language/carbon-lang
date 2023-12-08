// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

virtual namespace B

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1: ERROR: `namespace` declarations must end with a `;`.
// CHECK:STDERR: impl namespace
// CHECK:STDERR: ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:6: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: impl namespace
// CHECK:STDERR:      ^~~~~~~~~
impl namespace

// CHECK:STDOUT: - filename: fail_modifiers.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'impl', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'namespace', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
