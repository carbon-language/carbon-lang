// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/alias/fail_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/alias/fail_syntax.carbon

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:7: error: `alias` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: alias = b;
// CHECK:STDERR:       ^
// CHECK:STDERR:
alias = b;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:9: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a b;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias a b;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:10: error: expected expression [ExpectedExpr]
// CHECK:STDERR: alias a =;
// CHECK:STDERR:          ^
// CHECK:STDERR:
alias a =;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:8: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a;
// CHECK:STDERR:        ^
// CHECK:STDERR:
alias a;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:7: error: `alias` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: alias =;
// CHECK:STDERR:       ^
// CHECK:STDERR:
alias =;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:9: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a b c = d;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias a b c = d;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:13: error: `alias` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: alias a = b c d;
// CHECK:STDERR:             ^
// CHECK:STDERR:
alias a = b c d;

fn F() {
  // CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:8: error: `alias` introducer should be followed by a name [ExpectedDeclName]
  // CHECK:STDERR:   alias;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  alias;

  // CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   alias a =;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  alias a =;

  alias a =
// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

alias a = b

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+57]]:21: error: `alias` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_syntax.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:     {kind: 'Alias', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'Alias', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:         {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'Alias', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:         {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:       {kind: 'Alias', text: '=', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:       {kind: 'AliasIntroducer', text: 'alias'},
// CHECK:STDOUT:       {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:       {kind: 'AliasInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:     {kind: 'Alias', text: 'b', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
