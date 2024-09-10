// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/fail_todo_where_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/fail_todo_where_and.carbon

// TODO: Support `and` to allow multiple requirements in `where` expressions.

// CHECK:STDERR: fail_todo_where_and.carbon:[[@LINE+3]]:31: ERROR: Expected `,` or `)`.
// CHECK:STDERR: fn Foo(T: type where .U = i32 and .V == .W and .X impls I);
// CHECK:STDERR:                               ^~~
fn Foo(T: type where .U = i32 and .V == .W and .X impls I);

// CHECK:STDOUT: - filename: fail_todo_where_and.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'U'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
