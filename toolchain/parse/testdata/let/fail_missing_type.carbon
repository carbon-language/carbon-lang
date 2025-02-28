// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_missing_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_missing_type.carbon

// CHECK:STDERR: fail_missing_type.carbon:[[@LINE+4]]:7: error: expected `:` or `:!` in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: let a = 4;
// CHECK:STDERR:       ^
// CHECK:STDERR:
let a = 4;

// CHECK:STDOUT: - filename: fail_missing_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: 'a', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
