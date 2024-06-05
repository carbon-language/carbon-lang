// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/base_misplaced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/base_misplaced.carbon

// This is rejected by check.
base: i32;

// CHECK:STDOUT: - filename: base_misplaced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:       {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:       {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:     {kind: 'BaseDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
