// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/match_first.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/match_first.carbon

// --- empty.carbon

match_first {}

// --- fail_missing_braces.carbon

// CHECK:STDERR: fail_missing_braces.carbon:[[@LINE+4]]:12: error: expected `{` after `match_first` [ExpectedCurlyBraceAfter]
// CHECK:STDERR: match_first;
// CHECK:STDERR:            ^
// CHECK:STDERR:
match_first;

// --- fail_eof.carbon

match_first
// CHECK:STDERR: fail_eof.carbon:[[@LINE+4]]:1: error: expected `{` after `match_first` [ExpectedCurlyBraceAfter]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- with_impls.carbon

match_first {

impl forall [T: I] T as Interface {
  fn Add(self, b: Self) -> Self;
}

impl forall [T: J] T as Interface {
  fn Add(self, b: Self) -> Self;
}

}

// CHECK:STDOUT: - filename: empty.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'MatchFirstIntroducer', text: 'match_first'},
// CHECK:STDOUT:       {kind: 'MatchFirstDefinitionStart', text: '{', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'MatchFirst', text: '}', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_missing_braces.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'MatchFirstIntroducer', text: 'match_first'},
// CHECK:STDOUT:       {kind: 'MatchFirstDefinitionStart', text: 'match_first', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'MatchFirst', text: 'match_first', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'EmptyDecl', text: ';'},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_eof.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'MatchFirstIntroducer', text: 'match_first'},
// CHECK:STDOUT:       {kind: 'MatchFirstDefinitionStart', text: 'match_first', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'MatchFirst', text: 'match_first', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: with_impls.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'MatchFirstIntroducer', text: 'match_first'},
// CHECK:STDOUT:       {kind: 'MatchFirstDefinitionStart', text: '{', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'Forall', text: 'forall'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:               {kind: 'CompileTimeBindingPatternStart', text: ':', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:           {kind: 'ImplTypeAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameMaybeBeforeSignature', text: 'Add'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:             {kind: 'SelfBindingPattern', text: 'self', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:           {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'Forall', text: 'forall'},
// CHECK:STDOUT:             {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:               {kind: 'CompileTimeBindingPatternStart', text: ':', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'J'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ImplicitParamList', text: ']', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:           {kind: 'ImplTypeAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:         {kind: 'ImplDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameMaybeBeforeSignature', text: 'Add'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:             {kind: 'SelfBindingPattern', text: 'self', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'ImplDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'MatchFirst', text: '}', subtree_size: 55},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
