// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/fail_incomplete.carbon

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:12: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: constraint "Foo"
// CHECK:STDERR:            ^~~~~
// CHECK:STDERR:
constraint "Foo"

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:1: error: `constraint` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: constraint
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
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
