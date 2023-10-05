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
// CHECK:STDOUT:       {kind: 'Name', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'Name', text: 'Bar'},
// CHECK:STDOUT:       {kind: 'QualifiedDeclaration', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'Foo'},
// CHECK:STDOUT:             {kind: 'Name', text: 'Bar'},
// CHECK:STDOUT:           {kind: 'QualifiedDeclaration', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'Name', text: 'Baz'},
// CHECK:STDOUT:         {kind: 'QualifiedDeclaration', text: '.', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'Foo'},
// CHECK:STDOUT:                 {kind: 'Name', text: 'Bar'},
// CHECK:STDOUT:               {kind: 'MemberAccessExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'Name', text: 'Wiz'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpression', text: '.', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
