// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/fail_modifiers.carbon

virtual namespace B

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: error: `namespace` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: impl namespace
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:6: error: `namespace` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: impl namespace
// CHECK:STDERR:      ^~~~~~~~~
// CHECK:STDERR:
impl namespace

// CHECK:STDOUT: - filename: fail_modifiers.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'VirtualModifier', text: 'virtual'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'B'},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'impl', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'Namespace', text: 'namespace', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
