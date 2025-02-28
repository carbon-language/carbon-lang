// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration/fail_identifier_instead_of_sig.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration/fail_identifier_instead_of_sig.carbon

// CHECK:STDERR: fail_identifier_instead_of_sig.carbon:[[@LINE+4]]:8: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn foo bar;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
fn foo bar;

// CHECK:STDOUT: - filename: fail_identifier_instead_of_sig.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'foo'},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
