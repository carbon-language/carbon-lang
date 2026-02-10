// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/forall.carbon

impl forall [T:! type] T as Interface;

impl forall [T:! type, U:! Interface] U as Interface(T) {
}

// CHECK:STDOUT: - filename: forall.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'Forall', text: 'forall'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPatternStart', text: ':!', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:       {kind: 'ImplTypeAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'Forall', text: 'forall'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPatternStart', text: ':!', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'U'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPatternStart', text: ':!', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 11},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'U'},
// CHECK:STDOUT:         {kind: 'ImplTypeAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
