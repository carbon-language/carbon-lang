// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/alias/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/alias/basic.carbon

alias a = foo;
alias b = 0;
alias c = 1 + 3;

alias NS.ns = foo;

class C {
  alias c = foo;
}

fn F() {
  alias f = foo;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'c'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:       {kind: 'InfixOperatorPlus', text: '+', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'NS'},
// CHECK:STDOUT:       {kind: 'NameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'ns'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'c'},
// CHECK:STDOUT:         {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'Alias', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'f'},
// CHECK:STDOUT:         {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'Alias', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
