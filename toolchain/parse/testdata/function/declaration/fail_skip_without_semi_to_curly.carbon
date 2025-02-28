// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration/fail_skip_without_semi_to_curly.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration/fail_skip_without_semi_to_curly.carbon

// CHECK:STDERR: fail_skip_without_semi_to_curly.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: struct X { fn () }
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
struct X { fn () }
fn F();

// CHECK:STDOUT: - filename: fail_skip_without_semi_to_curly.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'InvalidParseStart', text: 'struct', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParseSubtree', text: '}', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
