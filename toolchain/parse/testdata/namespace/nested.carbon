// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace Foo;
namespace Foo.Bar;

fn Foo.Bar.Baz() {
  Foo.Bar.Wiz();
}

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:       {kind: 'QualifiedDecl', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:           {kind: 'QualifiedDecl', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'Baz'},
// CHECK:STDOUT:         {kind: 'QualifiedDecl', text: '.', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Foo'},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:               {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'Wiz'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
