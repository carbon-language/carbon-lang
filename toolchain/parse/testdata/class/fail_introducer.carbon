// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Improve the diagnostic to say that the `abstract` modifier can't be
// used on an `interface`.
// CHECK:STDERR: fail_introducer.carbon:[[@LINE+3]]:10: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: abstract interface I;
// CHECK:STDERR:          ^
abstract interface I;

// CHECK:STDERR: fail_introducer.carbon:[[@LINE+3]]:6: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: base fn F();
// CHECK:STDERR:      ^
base fn F();

// CHECK:STDERR: fail_introducer.carbon:[[@LINE+3]]:10: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: abstract base class C;
// CHECK:STDERR:          ^
abstract base class C;

// CHECK:STDOUT: - filename: fail_introducer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'interface', has_error: yes},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'fn', has_error: yes},
// CHECK:STDOUT:       {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'base', has_error: yes},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
