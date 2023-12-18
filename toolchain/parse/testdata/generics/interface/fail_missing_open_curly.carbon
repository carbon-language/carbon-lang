// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_missing_open_curly.carbon:[[@LINE+3]]:15: ERROR: `interface` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: interface Bar Baz {}
// CHECK:STDERR:               ^~~
interface Bar Baz {}

interface Foo

// CHECK:STDERR: fail_missing_open_curly.carbon:[[@LINE+13]]:21: ERROR: `interface` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_missing_open_curly.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: 'Foo', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
