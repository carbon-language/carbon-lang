// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:12: ERROR: `constraint` introducer should be followed by a name.
// CHECK:STDERR: constraint "Foo"
// CHECK:STDERR:            ^~~~~
constraint "Foo"

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:1: ERROR: `constraint` introducer should be followed by a name.
// CHECK:STDERR: constraint
// CHECK:STDERR: ^~~~~~~~~~
constraint

// CHECK:STDOUT: - filename: fail_incomplete.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '"Foo"', has_error: yes},
// CHECK:STDOUT:     {kind: 'NamedConstraintDecl', text: '"Foo"', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'NamedConstraintDecl', text: 'constraint', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
