// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/basic.carbon

namespace Foo;

fn Bar() {
  Foo.Baz();
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'NamespaceStart', text: 'namespace'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Namespace', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Bar'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Foo'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'Baz'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
