// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

impl forall [T:! type] T as Interface;

impl forall [T:! type, U:! Interface] U as Interface(T) {
}

// CHECK:STDOUT: - filename: forall.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplForall', text: 'forall', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:       {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'U'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:             {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ImplForall', text: 'forall', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'U'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
