// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/nested.carbon

namespace Foo;
namespace Foo.Bar;

fn Foo.Bar.Baz() {
  Foo.Bar.Wiz();
}

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Baz'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Foo'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'Bar'},
// CHECK:STDOUT:               {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'Wiz'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
