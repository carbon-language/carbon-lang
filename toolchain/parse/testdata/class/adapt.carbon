// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/adapt.carbon

// --- adapt.carbon

class A {
  adapt ();
}

class B {
  adapt A;
}

class C {
  adapt i32;
}

class D {
  extend adapt {.a: i32, .b: i32};
}

class E {
  adapt GetType(42);
}

// This is syntactically OK but semantically invalid.
adapt i32;

// --- fail_bad_syntax.carbon

class A {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+2]]:11-24: error: `adapt` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:
  adapt 1 trailing_token;
}

class B {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+2]]:8: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  adapt;
}

class C {
  adapt
// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+2]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
}

class D {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+2]]:9-13: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  adapt class;
}

fn F() {
  adapt i32;
}

// CHECK:STDOUT: - filename: adapt.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ │ ╭─StructTypeLiteralStart '{'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │ │ ├─StructTypeLiteralComma ','
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─StructTypeLiteralField ':'
// CHECK:STDOUT:   │ │ ├─StructTypeLiteral '}'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'E'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'GetType'
// CHECK:STDOUT:   │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '42'
// CHECK:STDOUT:   │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_bad_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ ├─AdaptDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ ├─AdaptDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ ├─AdaptDecl 'adapt' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─InvalidParse 'class' has_error
// CHECK:STDOUT:   │ ├─AdaptDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─AdaptDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
