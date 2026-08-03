// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/class.carbon

class C {
  impl as ForwardDeclared;
  impl C as ExplicitType {
  }
  impl Self as ExplicitSelf;
  extend impl as ExtendImpl {
    fn F();
  }
  extend impl C as InvalidButDiagnosedInCheck;
}

// CHECK:STDOUT: - filename: class.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ├─ImplDefaultSelfAs 'as'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'ForwardDeclared'
// CHECK:STDOUT:   │ ├─ImplDecl ';'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'ExplicitType'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   │ │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'ExplicitSelf'
// CHECK:STDOUT:   │ ├─ImplDecl ';'
// CHECK:STDOUT:   │ │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │   ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │   ├─ImplDefaultSelfAs 'as'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'ExtendImpl'
// CHECK:STDOUT:   │ │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ├─ImplDefinition '}'
// CHECK:STDOUT:   │ │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'InvalidButDiagnosedInCheck'
// CHECK:STDOUT:   │ ├─ImplDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
