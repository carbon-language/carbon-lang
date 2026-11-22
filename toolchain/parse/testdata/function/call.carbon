// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/call.carbon

// --- basic.carbon

fn F() {
  a.b.f(c.d, (e)).g();
}

// --- ref_args.carbon

fn F() {
  G(ref x);
  G(ref x, y);
  G(x, ref y);
}

// --- nested_ref.carbon

fn F() {
  G(x, (ref x,));
  G(x, (ref y, z));
  G(x, (y, ref z));

  G(x, {.y = ref y, .z = z});
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │               ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │               ├─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │             ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │             ├─IdentifierNameNotBeforeSignature 'f'
// CHECK:STDOUT:   │ │           ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │         ╭─CallExprStart '('
// CHECK:STDOUT:   │ │         │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │         │ ├─IdentifierNameNotBeforeSignature 'd'
// CHECK:STDOUT:   │ │         ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │         ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │         │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │         │ ├─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │         ├─ParenExpr ')'
// CHECK:STDOUT:   │ │       ╭─CallExpr ')'
// CHECK:STDOUT:   │ │       ├─IdentifierNameNotBeforeSignature 'g'
// CHECK:STDOUT:   │ │     ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: ref_args.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: nested_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │   │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │   │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │   │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ │ ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   │ │ ├─PrefixOperatorRef 'ref'
// CHECK:STDOUT:   │ │   │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │   │ ├─StructLiteralComma ','
// CHECK:STDOUT:   │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │   │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │   │ │ ├─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │   │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ │   ├─StructLiteral '}'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
