// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/fail_qualified_alternative_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/fail_qualified_alternative_name.carbon

choice X {
  // CHECK:STDERR: fail_qualified_alternative_name.carbon:[[@LINE+4]]:4: error: expected `,` or `}` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:   A.B
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  A.B
}

// CHECK:STDOUT: - filename: fail_qualified_alternative_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'X'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
