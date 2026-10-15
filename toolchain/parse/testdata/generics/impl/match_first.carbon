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

// CHECK:STDERR: fail_missing_braces.carbon:[[@LINE+2]]:12: error: expected `{` after `match_first` [ExpectedCurlyBraceAfter]
// CHECK:STDERR:
match_first;

// --- fail_eof.carbon

match_first
// CHECK:STDERR: fail_eof.carbon:[[@LINE+2]]:1: error: expected `{` after `match_first` [ExpectedCurlyBraceAfter]
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
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─MatchFirstIntroducer 'match_first'
// CHECK:STDOUT:   │ ╭─MatchFirstDefinitionStart '{'
// CHECK:STDOUT:   ├─MatchFirst '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_braces.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─MatchFirstIntroducer 'match_first'
// CHECK:STDOUT:   │ ╭─MatchFirstDefinitionStart 'match_first' has_error
// CHECK:STDOUT:   ├─MatchFirst 'match_first' has_error
// CHECK:STDOUT:   ├─EmptyDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_eof.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─MatchFirstIntroducer 'match_first'
// CHECK:STDOUT:   │ ╭─MatchFirstDefinitionStart 'match_first' has_error
// CHECK:STDOUT:   ├─MatchFirst 'match_first' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─MatchFirstIntroducer 'match_first'
// CHECK:STDOUT:   │ ╭─MatchFirstDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   ├─Forall 'forall'
// CHECK:STDOUT:   │ │   │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameMaybeBeforeSignature 'Add'
// CHECK:STDOUT:   │ │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   ├─Forall 'forall'
// CHECK:STDOUT:   │ │   │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IdentifierNameExpr 'J'
// CHECK:STDOUT:   │ │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameMaybeBeforeSignature 'Add'
// CHECK:STDOUT:   │ │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ├─SelfBindingPattern 'self'
// CHECK:STDOUT:   │ │ │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─ReturnType '->'
// CHECK:STDOUT:   │ │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─MatchFirst '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
