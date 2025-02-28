// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/member_access/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/member_access/simple.carbon

fn F(a: T) {
  a.x;
  a->x;
  a.x->y;
  a->x.y;
}

// CHECK:STDOUT: - filename: simple.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 29},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
