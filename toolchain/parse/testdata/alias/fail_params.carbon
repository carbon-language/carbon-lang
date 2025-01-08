// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/alias/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/alias/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+3]]:19: error: unexpected parameters after name declaration [UnexpectedParamsAfterDeclName]
// CHECK:STDERR: alias A(T:! type) = T*;
// CHECK:STDERR:                   ^
alias A(T:! type) = T*;

// CHECK:STDOUT: - filename: fail_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'A'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
