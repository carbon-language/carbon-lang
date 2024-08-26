// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/impl_where.carbon

interface Interface {
  let T:! type;
}

// CHECK:STDERR: impl_where.carbon:[[@LINE+8]]:29: ERROR: Expected expression.
// CHECK:STDERR: impl i32 as Interface where .T = () {
// CHECK:STDERR:                             ^
// CHECK:STDERR:
// CHECK:STDERR: impl_where.carbon:[[@LINE+4]]:29: ERROR: requirement should use `impls`, `=`, or `==` operator.
// CHECK:STDERR: impl i32 as Interface where .T = () {
// CHECK:STDERR:                             ^
// CHECK:STDERR:
impl i32 as Interface where .T = () {
}

// CHECK:STDERR: impl_where.carbon:[[@LINE+7]]:30: ERROR: Expected expression.
// CHECK:STDERR: impl bool as Interface where .T = i32;
// CHECK:STDERR:                              ^
// CHECK:STDERR:
// CHECK:STDERR: impl_where.carbon:[[@LINE+3]]:30: ERROR: requirement should use `impls`, `=`, or `==` operator.
// CHECK:STDERR: impl bool as Interface where .T = i32;
// CHECK:STDERR:                              ^
impl bool as Interface where .T = i32;

// CHECK:STDOUT: - filename: impl_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'WhereIntroducer', text: 'where', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: '}', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:       {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'WhereIntroducer', text: 'where', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
