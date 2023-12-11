// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+3]]:5: ERROR: Expected binding pattern.
// CHECK:STDERR: var package.val: i32;
// CHECK:STDERR:     ^~~~~~~
var package.val: i32;

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace package.NS;
// CHECK:STDERR:           ^~~~~~~
namespace package.NS;

// CHECK:STDERR: fail_in_name.carbon:[[@LINE+3]]:7: ERROR: `class` introducer should be followed by a name.
// CHECK:STDERR: class package.C {
// CHECK:STDERR:       ^~~~~~~
class package.C {
}

// CHECK:STDOUT: - filename: fail_in_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'package', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: 'package', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'package', has_error: yes},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
