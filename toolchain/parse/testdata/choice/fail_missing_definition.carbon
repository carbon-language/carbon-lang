// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_missing_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_missing_definition.carbon

// CHECK:STDERR: fail_missing_definition.carbon:[[@LINE+4]]:25: error: choice definition expected [ExpectedChoiceDefinition]
// CHECK:STDERR: choice MissingDefinition;
// CHECK:STDERR:                         ^
// CHECK:STDERR:
choice MissingDefinition;

// CHECK:STDOUT: - filename: fail_missing_definition.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'MissingDefinition'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
