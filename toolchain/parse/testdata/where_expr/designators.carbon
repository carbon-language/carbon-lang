// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/designators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/designators.carbon

// --- designator.carbon

fn Success() {
  .x;
  .Self;
}

// --- fail_designator.carbon

fn Fail() {
  // CHECK:STDERR: fail_designator.carbon:[[@LINE+4]]:4: ERROR: Expected identifier or `Self` after `.`.
  // CHECK:STDERR:   .1;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  .1;

  // CHECK:STDERR: fail_designator.carbon:[[@LINE+4]]:4: ERROR: Expected identifier or `Self` after `.`.
  // CHECK:STDERR:   .base;
  // CHECK:STDERR:    ^~~~
  // CHECK:STDERR:
  .base;

  // CHECK:STDERR: fail_designator.carbon:[[@LINE+4]]:4: ERROR: Expected identifier or `Self` after `.`.
  // CHECK:STDERR:   .;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  .;

  // CHECK:STDERR: fail_designator.carbon:[[@LINE+3]]:4: ERROR: Expected identifier or `Self` after `.`.
  // CHECK:STDERR:   .*;
  // CHECK:STDERR:    ^
  .*;
}

// --- fail_designator_eof.carbon

var x: i32 = .
// CHECK:STDOUT: - filename: designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Success'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'DotSelf', text: 'Self'},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Fail'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '1', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'base', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '*', has_error: yes},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PostfixOperatorStar', text: '*', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
