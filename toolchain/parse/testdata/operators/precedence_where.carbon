// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_where.carbon

fn PostfixStar(generic T: I* where .A* == .B*);

fn Ampersand(generic U: J & K where C & D impls E & F);

fn BinaryPlus(generic V: L + M where G & H impls N & O);

fn UnaryMinus(generic W: -P where -Q impls -R);

// CHECK:STDOUT: - filename: precedence_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'PostfixStar'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ │ │   ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │     ╭─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ │ │ │ │   ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'Ampersand'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'J'
// CHECK:STDOUT:   │ │ │ │     ├─IdentifierNameExpr 'K'
// CHECK:STDOUT:   │ │ │ │   ╭─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ │ │ │   ├─IdentifierNameExpr 'D'
// CHECK:STDOUT:   │ │ │ │ │ ╭─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'E'
// CHECK:STDOUT:   │ │ │ │ │ │ ├─IdentifierNameExpr 'F'
// CHECK:STDOUT:   │ │ │ │ │ ├─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'BinaryPlus'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'V'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'L'
// CHECK:STDOUT:   │ │ │ │     ├─IdentifierNameExpr 'M'
// CHECK:STDOUT:   │ │ │ │   ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameExpr 'G'
// CHECK:STDOUT:   │ │ │ │ │   ├─IdentifierNameExpr 'H'
// CHECK:STDOUT:   │ │ │ │ │ ╭─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'N'
// CHECK:STDOUT:   │ │ │ │ │ │ ├─IdentifierNameExpr 'O'
// CHECK:STDOUT:   │ │ │ │ │ ├─InfixOperatorAmp '&'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'UnaryMinus'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'W'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'P'
// CHECK:STDOUT:   │ │ │ │   ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameExpr 'Q'
// CHECK:STDOUT:   │ │ │ │ │ ╭─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'R'
// CHECK:STDOUT:   │ │ │ │ │ ├─PrefixOperatorMinus '-'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
