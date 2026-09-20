// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/require.carbon

// --- implicit_self_impls.carbon

interface Y {}

interface Z {
  require impls Y;
}

// --- explicit_self_impls.carbon

interface Y {}

interface Z {
  require Self impls Y;
}

// --- impls_where.carbon

interface Y { let Y1: type; }

interface Z {
  require impls Y where .Y1 = ();
}

// --- other_impls.carbon

interface Y {}

interface Z(T: type) {
  require T impls Y;
}

// --- extend.carbon

interface Y {}

interface Z {
  extend require impls Y;
}

// --- type_and.carbon

interface Y {}

interface Z {
  require impls Y & Y;
}

// --- fail_missing_impls.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_missing_impls.carbon:[[@LINE+2]]:12: error: expected `impls` in `require` declaration [RequireExpectedImpls]
  // CHECK:STDERR:
  require Y;
}

// --- fail_wrong_keyword.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_wrong_keyword.carbon:[[@LINE+2]]:11-12: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  require as Y;
}

// --- fail_as_in_self.carbon

interface Y {}

interface Z {
  // CHECK:STDERR: fail_as_in_self.carbon:[[@LINE+2]]:16-17: error: expected `impls` in `require` declaration [RequireExpectedImpls]
  // CHECK:STDERR:
  require Self as Y impls Y;
}

// CHECK:STDOUT: - filename: implicit_self_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─RequireDefaultSelfImpls 'impls'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: explicit_self_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ │ ╭─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ ├─RequireTypeImpls 'impls'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: impls_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'Y1'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─RequireDefaultSelfImpls 'impls'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'Y1'
// CHECK:STDOUT:   │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: other_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'Z'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │   │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │   │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ ├─RequireTypeImpls 'impls'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: extend.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─RequireDefaultSelfImpls 'impls'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: type_and.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─RequireDefaultSelfImpls 'impls'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │ ├─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ ├─RequireDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ ├─RequireDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_wrong_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─InvalidParse 'as' has_error
// CHECK:STDOUT:   │ ├─RequireDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_as_in_self.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─RequireIntroducer 'require'
// CHECK:STDOUT:   │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ ├─RequireDecl ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
