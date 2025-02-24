// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_unbraced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_unbraced.carbon

fn F() {
  if (a)
    // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+4]]:5: error: expected braced code block [ExpectedCodeBlock]
    // CHECK:STDERR:     if (b)
    // CHECK:STDERR:     ^~
    // CHECK:STDERR:
    if (b)
      // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+4]]:7: error: expected braced code block [ExpectedCodeBlock]
      // CHECK:STDERR:       if (c)
      // CHECK:STDERR:       ^~
      // CHECK:STDERR:
      if (c)
        // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+4]]:9: error: expected braced code block [ExpectedCodeBlock]
        // CHECK:STDERR:         d;
        // CHECK:STDERR:         ^
        // CHECK:STDERR:
        d;
}

// CHECK:STDOUT: - filename: fail_unbraced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:                   {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:                 {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:                     {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:                   {kind: 'ExprStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:               {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 14},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 16},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
