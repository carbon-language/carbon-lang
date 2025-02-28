// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_not.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_not.carbon

fn F() {
  not a and not b and not c;
}

// CHECK:STDOUT: - filename: precedence_not.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:                 {kind: 'PrefixOperatorNot', text: 'not', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperandAnd', text: 'and', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:               {kind: 'PrefixOperatorNot', text: 'not', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperatorAnd', text: 'and', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperandAnd', text: 'and', subtree_size: 7},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorNot', text: 'not', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ShortCircuitOperatorAnd', text: 'and', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
