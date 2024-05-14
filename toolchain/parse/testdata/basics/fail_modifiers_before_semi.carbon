// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers_before_semi.carbon:[[@LINE+3]]:18: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: protected virtual;
// CHECK:STDERR:                  ^
protected virtual;

// CHECK:STDOUT: - filename: fail_modifiers_before_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'ProtectedModifier', text: 'protected'},
// CHECK:STDOUT:       {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
