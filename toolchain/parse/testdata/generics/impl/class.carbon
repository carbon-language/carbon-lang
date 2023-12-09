// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  impl as ForwardDeclared;
  impl C as ExplicitType {
  }
  impl Self as ExplicitSelf;
  extend impl as ExtendImpl {
    fn F();
  }
  extend impl C as InvalidButDiagnosedInCheck;
}

// CHECK:STDOUT: - filename: class.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'ForwardDeclared'},
// CHECK:STDOUT:       {kind: 'ImplDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:           {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'ExplicitType'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'ExplicitSelf'},
// CHECK:STDOUT:       {kind: 'ImplDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:           {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'ExtendImpl'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:             {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:           {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'InvalidButDiagnosedInCheck'},
// CHECK:STDOUT:       {kind: 'ImplDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 36},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
