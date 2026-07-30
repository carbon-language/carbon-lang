// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/where_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/where_and.carbon

// --- success.carbon

fn OneAnd(generic T: type where .U = i32 and .V == .W);

fn TwoAnd(generic Y: I where .J impls K and .L == .M and .N = bool);

// --- fail_and_prefix.carbon

// CHECK:STDERR: fail_and_prefix.carbon:[[@LINE+4]]:36: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn AndPrefix(generic T: type where and .V == .W);
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR:
fn AndPrefix(generic T: type where and .V == .W);

// --- fail_and_suffix.carbon

// CHECK:STDERR: fail_and_suffix.carbon:[[@LINE+4]]:47: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn AndSuffix(generic Y: I where .J impls K and);
// CHECK:STDERR:                                               ^
// CHECK:STDERR:
fn AndSuffix(generic Y: I where .J impls K and);

// --- fail_and_early.carbon

// CHECK:STDERR: fail_and_early.carbon:[[@LINE+4]]:35: error: requirement should use `impls`, `=`, or `==` operator [ExpectedRequirementOperator]
// CHECK:STDERR: fn AndEarly(generic Z: L where .M and N);
// CHECK:STDERR:                                   ^~~
// CHECK:STDERR:
fn AndEarly(generic Z: L where .M and N);

// CHECK:STDOUT: - filename: success.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'OneAnd'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ │ │ ├─RequirementAnd 'and'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'V'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'W'
// CHECK:STDOUT:   │ │ │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'TwoAnd'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'J'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IdentifierNameExpr 'K'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementAnd 'and'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'L'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'M'
// CHECK:STDOUT:   │ │ │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ │ │ ├─RequirementAnd 'and'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'N'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_and_prefix.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'AndPrefix'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse 'and' has_error
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_and_suffix.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'AndSuffix'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'J'
// CHECK:STDOUT:   │ │ │ │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ │ │ ├─IdentifierNameExpr 'K'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │ │ ├─RequirementAnd 'and'
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_and_early.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'AndEarly'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameExpr 'L'
// CHECK:STDOUT:   │ │ │ │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'M'
// CHECK:STDOUT:   │ │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │ ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
