// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace 123
// CHECK:STDERR:           ^~~
namespace 123

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace ""
// CHECK:STDERR:           ^~
namespace ""

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace +
// CHECK:STDERR:           ^
namespace +

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace bool
// CHECK:STDERR:           ^~~~
namespace bool

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:11: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace namespace namespace namespace namespace
// CHECK:STDERR:           ^~~~~~~~~
namespace namespace namespace namespace namespace

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:1: ERROR: `namespace` introducer should be followed by a name.
// CHECK:STDERR: namespace
// CHECK:STDERR: ^~~~~~~~~
namespace

// CHECK:STDOUT: - filename: fail_incomplete.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '123', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: '123', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '""', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: '""', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '+', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: '+', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'bool', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'bool', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: 'namespace', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'namespace', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'namespace', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
