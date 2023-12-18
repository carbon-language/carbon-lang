// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;
base class B;
abstract class C;

class A {}
base class B {}
abstract class C {}

// CHECK:STDOUT: - filename: introducer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'BaseModifier', text: 'base'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'AbstractModifier', text: 'abstract'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
