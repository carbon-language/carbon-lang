// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/final_match_first.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/final_match_first.carbon

final match_first {

impl forall [T: I] T as Interface {
  fn Add(self, b: Self) -> Self;
}

impl forall [T: J] T as Interface {
  fn Add(self, b: Self) -> Self;
}

}

// CHECK:STDOUT: - filename: final_match_first.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─MatchFirstIntroducer 'match_first'
// CHECK:STDOUT:   │   ├─FinalModifier 'final'
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
