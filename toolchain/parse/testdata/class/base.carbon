// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class B {}

class D {
  base: B1;
  extend base: B2;
  base class Nested;
  private base class PrivateNested;
}

// CHECK:STDOUT: - filename: base.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'D'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:         {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'B1'},
// CHECK:STDOUT:       {kind: 'BaseDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:         {kind: 'ExtendModifier', text: 'extend'},
// CHECK:STDOUT:         {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'B2'},
// CHECK:STDOUT:       {kind: 'BaseDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Nested'},
// CHECK:STDOUT:       {kind: 'ClassDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'PrivateModifier', text: 'private'},
// CHECK:STDOUT:         {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'PrivateNested'},
// CHECK:STDOUT:       {kind: 'ClassDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
