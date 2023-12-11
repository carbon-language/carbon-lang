// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_missing_name.carbon:[[@LINE+3]]:11: ERROR: `interface` introducer should be followed by a name.
// CHECK:STDERR: interface {
// CHECK:STDERR:           ^
interface {
}

// CHECK:STDOUT: - filename: fail_missing_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '{', has_error: yes},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
