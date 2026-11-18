// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/fail_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/fail_rewrite.carbon

// Tests for reporting errors when a rewrite requirement doesn't follow
// `where .designator = rewrite`.

// --- fail_not_designator.carbon

// CHECK:STDERR:        ╭─┤ fail_not_designator.carbon:[[@LINE+9]]:44
// CHECK:STDERR: ->     │ fn NotDesignator(generic T: type where i32 = .U);
// CHECK:STDERR:        ·                                            ┳
// CHECK:STDERR:        ·                                            ┗━━┫ error: requirement can only use
// CHECK:STDERR:        ·                                               ┃        `=` after `.member`
// CHECK:STDERR:        ·                                               ┃        designator
// CHECK:STDERR:        ·                                               ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn NotDesignator(generic T: type where i32 = .U);

// --- fail_designator_in_parens.carbon

// CHECK:STDERR:        ╭─┤ fail_designator_in_parens.carbon:[[@LINE+9]]:47
// CHECK:STDERR: ->     │ fn DesignatorInParens(generic V: I where (.J) = bool);
// CHECK:STDERR:        ·                                               ┳
// CHECK:STDERR:        ·                                               ┗━━┫ error: requirement can only
// CHECK:STDERR:        ·                                                  ┃        use `=` after
// CHECK:STDERR:        ·                                                  ┃        `.member` designator
// CHECK:STDERR:        ·                                                  ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn DesignatorInParens(generic V: I where (.J) = bool);

// --- fail_dot_self.carbon

// CHECK:STDERR:        ╭─┤ fail_dot_self.carbon:[[@LINE+8]]:37
// CHECK:STDERR: ->     │ fn DotSelf(generic W: K where .Self = f32);
// CHECK:STDERR:        ·                                     ┳
// CHECK:STDERR:        ·                                     ┗━━┫ error: requirement can only use `=`
// CHECK:STDERR:        ·                                        ┃        after `.member` designator
// CHECK:STDERR:        ·                                        ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn DotSelf(generic W: K where .Self = f32);

// --- fail_dot_keyword.carbon

// CHECK:STDERR:        ╭─┤ fail_dot_keyword.carbon:[[@LINE+10]]:35
// CHECK:STDERR: ->     │ fn DotKeyword(generic W: K where .and = u8);
// CHECK:STDERR:        ·                                   ┳━━
// CHECK:STDERR:        ·                                   ┗━━┫ error: expected identifier or `Self`
// CHECK:STDERR:        ·                                      ┃        after `.`; prefix reserved word
// CHECK:STDERR:        ·                                      ┃        with `r#` to form a valid
// CHECK:STDERR:        ·                                      ┃        identifier
// CHECK:STDERR:        ·                                      ┃        [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn DotKeyword(generic W: K where .and = u8);

// --- fail_postfix_after_designator.carbon

// CHECK:STDERR:        ╭─┤ fail_postfix_after_designator.carbon:[[@LINE+10]]:50
// CHECK:STDERR: ->     │ fn PostfixAfterDesignator(generic X: L where .M* = u64*);
// CHECK:STDERR:        ·                                                  ┳
// CHECK:STDERR:        ·                                                  ┗━━┫ error: requirement can
// CHECK:STDERR:        ·                                                     ┃        only use `=` after
// CHECK:STDERR:        ·                                                     ┃        `.member`
// CHECK:STDERR:        ·                                                     ┃        designator
// CHECK:STDERR:        ·                                                     ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn PostfixAfterDesignator(generic X: L where .M* = u64*);

// --- fail_binary_op_after_designator.carbon

// CHECK:STDERR:        ╭─┤ fail_binary_op_after_designator.carbon:[[@LINE+11]]:55
// CHECK:STDERR: ->     │ fn BinaryOpAfterDesignator(generic Y: N where .O + .P = {});
// CHECK:STDERR:        ·                                                       ┳
// CHECK:STDERR:        ·                                                       ┗━━┫ error: requirement
// CHECK:STDERR:        ·                                                          ┃        can only use
// CHECK:STDERR:        ·                                                          ┃        `=` after
// CHECK:STDERR:        ·                                                          ┃        `.member`
// CHECK:STDERR:        ·                                                          ┃        designator
// CHECK:STDERR:        ·                                                          ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn BinaryOpAfterDesignator(generic Y: N where .O + .P = {});

// --- fail_after_and.carbon

// CHECK:STDERR:        ╭─┤ fail_after_and.carbon:[[@LINE+10]]:50
// CHECK:STDERR: ->     │ fn AfterAnd(generic Z: Q where .R impls S and () = .A);
// CHECK:STDERR:        ·                                                  ┳
// CHECK:STDERR:        ·                                                  ┗━━┫ error: requirement can
// CHECK:STDERR:        ·                                                     ┃        only use `=` after
// CHECK:STDERR:        ·                                                     ┃        `.member`
// CHECK:STDERR:        ·                                                     ┃        designator
// CHECK:STDERR:        ·                                                     ┃        [RequirementEqualAfterNonDesignator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn AfterAnd(generic Z: Q where .R impls S and () = .A);

// CHECK:STDOUT: - filename: fail_not_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'NotDesignator'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'U'
// CHECK:STDOUT:   │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_designator_in_parens.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'DesignatorInParens'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'V'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │ │   │ │ │ ╭─IdentifierNameNotBeforeSignature 'J'
// CHECK:STDOUT:   │ │ │   │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   │ ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_dot_self.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'DotSelf'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'W'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'K'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │ ╭─SelfTypeName 'Self'
// CHECK:STDOUT:   │ │ │   │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ ├─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_dot_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'DotKeyword'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'W'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'K'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'and' has_error
// CHECK:STDOUT:   │ │ │   │ ├─DesignatorExpr '.' has_error
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ ├─UnsignedIntTypeLiteral 'u8'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_postfix_after_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'PostfixAfterDesignator'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'X'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'L'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'M'
// CHECK:STDOUT:   │ │ │   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ │ ╭─UnsignedIntTypeLiteral 'u64'
// CHECK:STDOUT:   │ │ │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_binary_op_after_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'BinaryOpAfterDesignator'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'N'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'O'
// CHECK:STDOUT:   │ │ │   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   │ │ │ ╭─IdentifierNameNotBeforeSignature 'P'
// CHECK:STDOUT:   │ │ │   │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   │ ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │ ├─StructLiteral '}'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_after_and.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'AfterAnd'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'Z'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameExpr 'Q'
// CHECK:STDOUT:   │ │ │   │ ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │ │   │ │   ╭─IdentifierNameNotBeforeSignature 'R'
// CHECK:STDOUT:   │ │ │   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ │   │ │ ├─IdentifierNameExpr 'S'
// CHECK:STDOUT:   │ │ │   │ ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ │   │ ├─RequirementAnd 'and'
// CHECK:STDOUT:   │ │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │   ├─WhereExpr 'where' has_error
// CHECK:STDOUT:   │ │ │ ╭─CompileTimeBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ │ ├─DefaultValueExprStart '='
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ │ │ ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ ├─DefaultValuePattern '=' has_error
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
