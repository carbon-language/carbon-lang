// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/while/fail_unbraced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/while/fail_unbraced.carbon

fn F() {
  while (a)
    // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+4]]:5: error: expected braced code block [ExpectedCodeBlock]
    // CHECK:STDERR:     break;
    // CHECK:STDERR:     ^~~~~
    // CHECK:STDERR:
    break;
}

// CHECK:STDOUT: - filename: fail_unbraced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhileConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'WhileCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'break', has_error: yes},
// CHECK:STDOUT:             {kind: 'BreakStatementStart', text: 'break'},
// CHECK:STDOUT:           {kind: 'BreakStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'break', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'WhileStatement', text: 'while', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
