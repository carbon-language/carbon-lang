// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/member_access/fail_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/member_access/fail_keyword.carbon

fn F() {
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   a.self;
  // CHECK:STDERR:     ^~~~
  // CHECK:STDERR:
  a.self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   a.Self;
  // CHECK:STDERR:     ^~~~
  // CHECK:STDERR:
  a.Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: error: expected identifier after `.` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   a.for;
  // CHECK:STDERR:     ^~~
  // CHECK:STDERR:
  a.for;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:6: error: expected identifier after `->` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   p->self;
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  p->self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:6: error: expected identifier after `->` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   p->Self;
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  p->Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:6: error: expected identifier after `->` [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:   p->while;
  // CHECK:STDERR:      ^~~~~
  // CHECK:STDERR:
  p->while;
}

// CHECK:STDOUT: - filename: fail_keyword.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'self', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'for', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'self', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'while', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 30},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
