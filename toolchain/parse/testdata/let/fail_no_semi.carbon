// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_no_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_no_semi.carbon

let

// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+17]]:21: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_semi.carbon:[[@LINE+13]]:21: error: `let` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: '', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: '', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: 'let', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
