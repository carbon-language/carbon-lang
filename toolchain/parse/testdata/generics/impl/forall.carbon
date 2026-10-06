// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/forall.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/forall.carbon

impl forall [T: type] T as Interface;

impl forall [T: type, U: Interface] U as Interface(T) {
}

// CHECK:STDOUT: - filename: forall.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   ├─ImplDecl ';'
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   ├─Forall 'forall'
// CHECK:STDOUT:   │   │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │   ├─ImplicitParamList ']'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'U'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   │   ╭─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   │   │ ╭─CallExprStart '('
// CHECK:STDOUT:   │   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │   ├─CallExpr ')'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
