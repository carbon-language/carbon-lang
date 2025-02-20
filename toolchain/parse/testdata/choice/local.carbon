// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/local.carbon

fn F() {
  choice C {
    X,
    Y
  }
}

// CHECK:STDOUT: - filename: local.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:         {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'X'},
// CHECK:STDOUT:         {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:       {kind: 'ChoiceDefinition', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
