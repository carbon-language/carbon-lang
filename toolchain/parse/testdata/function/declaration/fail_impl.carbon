// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:1: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: impl foo bar;
// CHECK:STDERR: ^~~~
impl foo bar;
// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:1: ERROR: Unrecognized declaration introducer.
// CHECK:STDERR: impl
// CHECK:STDERR: ^~~~
impl

// CHECK:STDOUT: - filename: fail_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'impl', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'impl', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: 'impl', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
