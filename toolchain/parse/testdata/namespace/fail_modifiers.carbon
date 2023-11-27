// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:37: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:                                     ^~~~~~~~~
private abstract base default final namespace Foo;

// CHECK:STDOUT: - filename: fail_modifiers.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'namespace', has_error: yes},
// CHECK:STDOUT:       {kind: 'AccessModifierKeyword', text: 'private'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'base'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'default'},
// CHECK:STDOUT:       {kind: 'DeclModifierKeyword', text: 'final'},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
