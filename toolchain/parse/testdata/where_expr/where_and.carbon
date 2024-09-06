// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/where_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/where_and.carbon

fn OneAnd(T:! type where .U = i32 and .V == .W);

fn TwoAnd(Y:! I where .J impls K and .L == .M and .N = bool);

// CHECK:STDOUT: - filename: where_and.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'OneAnd'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereIntroducer', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'U'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'RequirementAssign', text: '=', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'V'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'W'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEquals', text: '==', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'TwoAnd'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'Y'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:             {kind: 'WhereIntroducer', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'J'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'L'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'M'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEquals', text: '==', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'N'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'RequirementAssign', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 18},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
