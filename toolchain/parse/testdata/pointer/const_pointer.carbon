// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/const_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/const_pointer.carbon

fn A() -> const i32* { return A(); }
fn B() -> (const i32)* { return B(); }
fn C() -> const (i32*) { return C(); }

// CHECK:STDOUT: - filename: const_pointer.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'A'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │     ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │   ╭─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │   │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'B'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │     ╭─ParenExprStart '('
// CHECK:STDOUT:   │   │     │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │     ├─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │   │   ╭─ParenExpr ')'
// CHECK:STDOUT:   │   │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'C'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │     ╭─ParenExprStart '('
// CHECK:STDOUT:   │   │     │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │     ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │   │   ╭─ParenExpr ')'
// CHECK:STDOUT:   │   │ ╭─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
