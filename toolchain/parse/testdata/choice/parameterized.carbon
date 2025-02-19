// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/choice/parameterized.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/choice/parameterized.carbon

choice OptionalElement(T:! type) {
  Element(value: T),
  None,
}

// CHECK:STDOUT: - filename: parameterized.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'OptionalElement'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'Element'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'value'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'None'},
// CHECK:STDOUT:       {kind: 'ChoiceAlternativeListComma', text: ','},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
