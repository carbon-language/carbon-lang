// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

// Enum values for ParserState, listing states for Parser.
//
// Each cluster of enum values has a comment explaining their use and possible
// resulting state stacks. In a numbered state stack, "1" will be the first
// state processed; in other words, the first popped off the stack after the
// current state completes.
//
// Enum value clusters will have the form `XAsY`, where the `X` half indicates
// common handling (typically `HandleX` on Parser) and the second half indicates
// a processing nuance for the common handling.
//
// This is an X-macro; the argument should be a macro taking a single argument,
// the name.
#define CARBON_PARSER_STATE(X)                                                 \
  /* Handles the `{` of a brace expression.                                    \
   *                                                                           \
   * If `CloseCurlyBrace`:                                                     \
   *   1. BraceExpressionFinishAsUnknown                                       \
   * Else:                                                                     \
   *   1. BraceExpressionParameterAsUnknown                                    \
   *   2. BraceExpressionFinishAsUnknown                                       \
   */                                                                          \
  X(BraceExpression)                                                           \
                                                                               \
  /* Handles a brace expression parameter. Note this will always start as      \
   * unknown, but should be known after the first valid parameter. All later   \
   * inconsistent parameters are invalid.                                      \
   *                                                                           \
   * If valid:                                                                 \
   *   1. DesignatorExpressionAsStruct                                         \
   *   2. BraceExpressionParameterAfterDesignatorAs(Type|Value|Unknown)        \
   * Else:                                                                     \
   *   1. BraceExpressionParameterFinishAs(Type|Value|Unknown)                 \
   */                                                                          \
  X(BraceExpressionParameterAsType)                                            \
  X(BraceExpressionParameterAsValue)                                           \
  X(BraceExpressionParameterAsUnknown)                                         \
                                                                               \
  /* Handles a brace expression parameter after the initial designator. This   \
   * should be at a `:` or `=`, depending on whether it's a type or value      \
   * literal.                                                                  \
   *                                                                           \
   * If valid:                                                                 \
   *   1. Expression                                                           \
   *   2. BraceExpressionParameterFinishAs(Type|Value|Unknown)                 \
   * Else:                                                                     \
   *   1. BraceExpressionParameterFinishAs(Type|Value|Unknown)                 \
   */                                                                          \
  X(BraceExpressionParameterAfterDesignatorAsType)                             \
  X(BraceExpressionParameterAfterDesignatorAsValue)                            \
  X(BraceExpressionParameterAfterDesignatorAsUnknown)                          \
                                                                               \
  /* Handles the end of a brace expression parameter.                          \
   *                                                                           \
   * If `Comma`:                                                               \
   *   1. BraceExpressionParameterAsUnknown                                    \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(BraceExpressionParameterFinishAsType)                                      \
  X(BraceExpressionParameterFinishAsValue)                                     \
  X(BraceExpressionParameterFinishAsUnknown)                                   \
                                                                               \
  /* Handles the `}` of a brace expression.                                    \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(BraceExpressionFinishAsType)                                               \
  X(BraceExpressionFinishAsValue)                                              \
  X(BraceExpressionFinishAsUnknown)                                            \
                                                                               \
  /* Handles a call expression `(...)`.                                        \
   *                                                                           \
   * If `CloseParen`:                                                          \
   *   1. CallExpressionFinish                                                 \
   * Else:                                                                     \
   *   1. Expression                                                           \
   *   2. CallExpressionParameterFinish                                        \
   *   3. CallExpressionFinish                                                 \
   */                                                                          \
  X(CallExpression)                                                            \
                                                                               \
  /* Handles the `,` or `)` after a call parameter.                            \
   *                                                                           \
   * If `Comma`:                                                               \
   *   1. Expression                                                           \
   *   2. CallExpressionParameterFinish                                        \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(CallExpressionParameterFinish)                                             \
                                                                               \
  /* Handles finishing the call expression.                                    \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(CallExpressionFinish)                                                      \
                                                                               \
  /* Handles processing at the `{` on a typical code block.                    \
   *                                                                           \
   * If `OpenCurlyBrace`:                                                      \
   *   1. StatementScopeLoop                                                   \
   *   2. CodeBlockFinish                                                      \
   * Else:                                                                     \
   *   1. Statement                                                            \
   *   2. CodeBlockFinish                                                      \
   */                                                                          \
  X(CodeBlock)                                                                 \
                                                                               \
  /* Handles processing at the `}` on a typical code block, after a statement  \
   * scope is done.                                                            \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(CodeBlockFinish)                                                           \
                                                                               \
  /* Handles processing of a declaration scope. Things like fn, class,         \
   * interface, and so on.                                                     \
   *                                                                           \
   * If `EndOfFile`:                                                           \
   *   (state done)                                                            \
   * If `Fn`:                                                                  \
   *   1. FunctionIntroducer                                                   \
   *   2. DeclarationLoop                                                      \
   * If `Package`:                                                             \
   *   1. Package                                                              \
   *   2. DeclarationLoop                                                      \
   * If `Semi`:                                                                \
   *   1. DeclarationLoop                                                      \
   * If `Var`:                                                                 \
   *   1. Var                                                                  \
   *   2. DeclarationLoop                                                      \
   * If `interface`:                                                           \
   *   1. InterfaceIntroducer                                                  \
   *   2. DeclarationLoop                                                      \
   * Else:                                                                     \
   *   1. DeclarationLoop                                                      \
   */                                                                          \
  X(DeclarationLoop)                                                           \
                                                                               \
  /* Handles a designator expression, such as `.z` in `x.(y.z)`.               \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(DesignatorAsExpression)                                                    \
  X(DesignatorAsStruct)                                                        \
                                                                               \
  /* Handles processing of an expression.                                      \
   *                                                                           \
   * If valid prefix operator:                                                 \
   *   1. Expression                                                           \
   *   2. ExpressionLoopForPrefix                                              \
   * Else:                                                                     \
   *   1. ExpressionInPostfix                                                  \
   *   2. ExpressionLoop                                                       \
   */                                                                          \
  X(Expression)                                                                \
                                                                               \
  /* Handles the initial part of postfix expressions, such as an identifier or \
   * literal value, then proceeds to the loop.                                 \
   *                                                                           \
   * If `Identifier` or literal (including type literals):                     \
   *   1. ExpressionInPostfixLoop                                              \
   * If `OpenCurlyBrace`:                                                      \
   *   1. BraceExpression                                                      \
   *   2. ExpressionInPostfixLoop                                              \
   * If `OpenParen`:                                                           \
   *   1. ParenExpression                                                      \
   *   2. ExpressionInPostfixLoop                                              \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(ExpressionInPostfix)                                                       \
                                                                               \
  /* Handles looping through elements following the initial postfix            \
   * expression, such as designators or parenthesized parameters.              \
   *                                                                           \
   * If `Period`:                                                              \
   *   1. DesignatorAsExpression                                               \
   *   2. ExpressionInPostfixLoop                                              \
   * If `OpenParen`:                                                           \
   *   1. CallExpression                                                       \
   *   2. ExpressionInPostfixLoop                                              \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(ExpressionInPostfixLoop)                                                   \
                                                                               \
  /* Handles processing of an expression.                                      \
   *                                                                           \
   * If binary operator:                                                       \
   *   1. Expression                                                           \
   *   2. ExpressionLoopForBinary                                              \
   * If postfix operator:                                                      \
   *   1. ExpressionLoop                                                       \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(ExpressionLoop)                                                            \
                                                                               \
  /* Completes an ExpressionLoop pass by adding an infix operator, then goes   \
   * back to ExpressionLoop.                                                   \
   *                                                                           \
   * Always:                                                                   \
   *   1. ExpressionLoop                                                       \
   */                                                                          \
  X(ExpressionLoopForBinary)                                                   \
                                                                               \
  /* Completes an ExpressionLoop pass by adding a prefix operator, then goes   \
   * back to ExpressionLoop.                                                   \
   *                                                                           \
   * Always:                                                                   \
   *   1. ExpressionLoop                                                       \
   */                                                                          \
  X(ExpressionLoopForPrefix)                                                   \
                                                                               \
  /* Handles the `;` for an expression statement, which is different from most \
   * keyword statements.                                                       \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(ExpressionStatementFinish)                                                 \
                                                                               \
  /* Handles processing of a function's `fn <name>(`, and enqueues parameter   \
   * list handling.                                                            \
   *                                                                           \
   * If invalid:                                                               \
   *   (state done)                                                            \
   * If parenthesized parameters:                                              \
   *   1. PatternAsFunctionParameter                                           \
   *   2. FunctionParameterListFinish                                          \
   *   3. FunctionAfterParameterList                                           \
   * Else:                                                                     \
   *   1. FunctionParameterListFinish                                          \
   *   2. FunctionAfterParameterList                                           \
   */                                                                          \
  X(FunctionIntroducer)                                                        \
                                                                               \
  /* Starts function parameter processing.                                     \
   *                                                                           \
   * Always:                                                                   \
   *   1. PatternAsFunctionParameter                                           \
   *   2. FunctionParameterFinish                                              \
   */                                                                          \
  X(FunctionParameter)                                                         \
                                                                               \
  /* Finishes function parameter processing, including `,`. If there are more  \
   * parameters, enqueues another parameter processing state.                  \
   *                                                                           \
   * If `Comma` without `CloseParen`:                                          \
   *   1. FunctionParameter                                                    \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(FunctionParameterFinish)                                                   \
                                                                               \
  /* Handles processing of a function's parameter list `)`.                    \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(FunctionParameterListFinish)                                               \
                                                                               \
  /* Handles processing of a function's syntax after `)`, primarily the        \
   * possibility a `->` return type is there. Always enqueues signature finish \
   * handling.                                                                 \
   *                                                                           \
   * If `MinusGreater`:                                                        \
   *   1. Expression                                                           \
   *   2. FunctionReturnTypeFinish                                             \
   *   3. FunctionSignatureFinish                                              \
   * Else:                                                                     \
   *   1. FunctionSignatureFinish                                              \
   */                                                                          \
  X(FunctionAfterParameterList)                                                \
                                                                               \
  /* Finishes a function return type.                                          \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(FunctionReturnTypeFinish)                                                  \
                                                                               \
  /* Finishes a function signature. If it's a declaration, the function is     \
   * done; otherwise, this also starts definition processing.                  \
   *                                                                           \
   * If `Semi`:                                                                \
   *   (state done)                                                            \
   * If `OpenCurlyBrace`:                                                      \
   *   1. StatementScopeLoop                                                   \
   *   2. FunctionDefinitionFinish                                             \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(FunctionSignatureFinish)                                                   \
                                                                               \
  /* Finishes a function definition.                                           \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(FunctionDefinitionFinish)                                                  \
                                                                               \
  /* Finishes an interface definition.                                         \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(InterfaceDefinitionFinish)                                                 \
                                                                               \
  /* Handles parsing the body of an interface.                                 \
   *                                                                           \
   * If `}`:                                                                   \
   *   (state done)                                                            \
   * Else:                                                                     \
   *   1. InterfaceDefinitionLoop                                              \
   */                                                                          \
  X(InterfaceDefinitionLoop)                                                   \
                                                                               \
  /* Handles processing of a intefaces's `interface <name> {`.                 \
   *                                                                           \
   * If invalid:                                                               \
   *   1. InterfaceDefinitionFinish                                            \
   * If `{` is missing:                                                        \
   *   1. InterfaceDefinitionFinish                                            \
   * Else:                                                                     \
   *   1. InterfaceDefinitionLoop                                              \
   *   2. InterfaceDefinitionFinish                                            \
   */                                                                          \
  X(InterfaceIntroducer)                                                       \
                                                                               \
  /* Handles `package`.                                                        \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(Package)                                                                   \
                                                                               \
  /* Handles the processing of a `(condition)` up through the expression.      \
   *                                                                           \
   * Always:                                                                   \
   *   1. Expression                                                           \
   *   2. ParenConditionAs(If|While)Finish                                     \
   */                                                                          \
  X(ParenConditionAsIf)                                                        \
  X(ParenConditionAsWhile)                                                     \
                                                                               \
  /* Finishes the processing of a `(condition)` after the expression.          \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(ParenConditionFinishAsIf)                                                  \
  X(ParenConditionFinishAsWhile)                                               \
                                                                               \
  /* Handles the `(` of a parenthesized expression.                            \
   *                                                                           \
   * If `CloseParen`:                                                          \
   *   1. ParenExpressionFinishAsTuple                                         \
   * Else:                                                                     \
   *   1. Expression                                                           \
   *   2. ParenExpressionParameterFinishAsUnknown                              \
   *   3. ParenExpressionFinish                                                \
   */                                                                          \
  X(ParenExpression)                                                           \
                                                                               \
  /* Handles the end of a parenthesized expression's parameter. This will      \
   * start as AsUnknown on the first parameter; if there are more, it switches \
   * to AsTuple processing.                                                    \
   *                                                                           \
   * If `Comma` without `CloseParen`:                                          \
   *   1. Expression                                                           \
   *   2. ParenExpressionParameterFinishAsTuple                                \
   *   SPECIAL: Parent becomes ParenExpressionFinishAsTuple                    \
   * If `Comma` with `CloseParen`:                                             \
   *   (state done)                                                            \
   *   SPECIAL: Parent becomes ParenExpressionFinishAsTuple                    \
   * Else `CloseParen`:                                                        \
   *   (state done)                                                            \
   */                                                                          \
  X(ParenExpressionParameterFinishAsUnknown)                                   \
  X(ParenExpressionParameterFinishAsTuple)                                     \
                                                                               \
  /* Handles the `)` of a parenthesized expression.                            \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(ParenExpressionFinish)                                                     \
  X(ParenExpressionFinishAsTuple)                                              \
                                                                               \
  /* Handles pattern parsing for a pattern, enqueuing type expression          \
   * processing. This covers function parameter and `var` support.             \
   *                                                                           \
   * If valid:                                                                 \
   *   1. Expression                                                           \
   *   2. PatternFinish                                                        \
   * Else:                                                                     \
   *   1. PatternFinish                                                        \
   */                                                                          \
  X(PatternAsFunctionParameter)                                                \
  X(PatternAsVariable)                                                         \
                                                                               \
  /* Finishes pattern processing.                                              \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(PatternFinish)                                                             \
                                                                               \
  /* Handles a single statement. While typically within a statement block,     \
   * this can also be used for error recovery where we expect a statement      \
   * block and are missing braces.                                             \
   *                                                                           \
   * If `Break`:                                                               \
   *   1. StatementBreakFinish                                                 \
   *   (state done)                                                            \
   * If `Continue`:                                                            \
   *   1. StatementContinueFinish                                              \
   *   (state done)                                                            \
   * If `For`:                                                                 \
   *   1. StatementForHeader                                                   \
   *   2. StatementForFinish                                                   \
   * If `If`:                                                                  \
   *   1. StatementIf                                                          \
   * If `Return`:                                                              \
   *   1. StatementReturn                                                      \
   * If `Var`:                                                                 \
   *   1. VarAsSemicolon                                                       \
   * If `While`:                                                               \
   *   1. StatementWhile                                                       \
   * Else:                                                                     \
   *   1. Expression                                                           \
   *   2. ExpressionStatementFinish                                            \
   */                                                                          \
  X(Statement)                                                                 \
                                                                               \
  /* Handles `break` processing at the `;`.                                    \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementBreakFinish)                                                      \
                                                                               \
  /* Handles `continue` processing at the `;`.                                 \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementContinueFinish)                                                   \
                                                                               \
  /* Handles `for` processing of `(var`, proceeding to a pattern before        \
   * continuing.                                                               \
   *                                                                           \
   * If no `OpenParen`:                                                        \
   *   1. CodeBlock                                                            \
   * If `Var`:                                                                 \
   *   1. VarAsFor                                                             \
   *   2. StatementForHeaderIn                                                 \
   * Else:                                                                     \
   *   1. StatementForHeaderIn                                                 \
   */                                                                          \
  X(StatementForHeader)                                                        \
                                                                               \
  /* Handles `for` procesisng of `in`, proceeding to an expression before      \
   * continuing.                                                               \
   *                                                                           \
   * If `In` or `Colon`:                                                       \
   *   1. Expression                                                           \
   *   2. StatementForHeaderFinish                                             \
   * Else:                                                                     \
   *   1. StatementForHeaderFinish                                             \
   */                                                                          \
  X(StatementForHeaderIn)                                                      \
                                                                               \
  /* Handles `for` processing of `)`, proceeding to the statement block.       \
   *                                                                           \
   * Always:                                                                   \
   *   1. CodeBlock                                                            \
   */                                                                          \
  X(StatementForHeaderFinish)                                                  \
                                                                               \
  /* Handles `for` processing at the final `}`.                                \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementForFinish)                                                        \
                                                                               \
  /* Handles `if` processing at the start.                                     \
   *                                                                           \
   * Always:                                                                   \
   *   1. ParenConditionAsIf                                                   \
   *   2. StatementIfConditionFinish                                           \
   */                                                                          \
  X(StatementIf)                                                               \
                                                                               \
  /* Handles `if` processing between the condition and start of the first code \
   * block.                                                                    \
   *                                                                           \
   * Always:                                                                   \
   *   1. CodeBlock                                                            \
   *   2. StatementIfThenBlockFinish                                           \
   */                                                                          \
  X(StatementIfConditionFinish)                                                \
                                                                               \
  /* Handles `if` processing after the end of the first code block, with the   \
   * optional `else`.                                                          \
   *                                                                           \
   * If `Else` then `If`:                                                      \
   *   1. CodeBlock                                                            \
   *   2. StatementIfElseBlockFinish                                           \
   * If `Else`:                                                                \
   *   1. StatementIf                                                          \
   *   2. StatementIfElseBlockFinish                                           \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementIfThenBlockFinish)                                                \
                                                                               \
  /* Handles `if` processing after a provided `else` code block.               \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementIfElseBlockFinish)                                                \
                                                                               \
  /* Handles `return` processing.                                              \
   *                                                                           \
   * If `Semi`:                                                                \
   *   1. StatementReturnFinish                                                \
   * Else:                                                                     \
   *   1. Expression                                                           \
   *   2. StatementReturnFinish                                                \
   */                                                                          \
  X(StatementReturn)                                                           \
                                                                               \
  /* Handles `return` processing at the `;` when there's an expression.        \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementReturnFinish)                                                     \
                                                                               \
  /* Handles processing of statements within a scope.                          \
   *                                                                           \
   * If `CloseCurlyBrace`:                                                     \
   *   (state done)                                                            \
   * Else:                                                                     \
   *   1. Statement                                                            \
   *   2. StatementScopeLoop                                                   \
   */                                                                          \
  X(StatementScopeLoop)                                                        \
                                                                               \
  /* Handles `while` processing.                                               \
   *                                                                           \
   * Always:                                                                   \
   *   1. ParenConditionAsWhile                                                \
   *   2. StatementWhileConditionFinish                                        \
   */                                                                          \
  X(StatementWhile)                                                            \
                                                                               \
  /* Handles `while` processing between the condition and start of the code    \
   * block.                                                                    \
   *                                                                           \
   * Always:                                                                   \
   *   1. CodeBlock                                                            \
   *   2. StatementWhileBlockFinish                                            \
   */                                                                          \
  X(StatementWhileConditionFinish)                                             \
                                                                               \
  /* Handles `while` processing after the end of the code block.               \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(StatementWhileBlockFinish)                                                 \
                                                                               \
  /* Handles the start of a `var`.                                             \
   *                                                                           \
   * Always:                                                                   \
   *   1. PatternAsVariable                                                    \
   *   2. VarAfterPattern                                                      \
   *   3. VarFinishAs(Semicolon|For)                                           \
   */                                                                          \
  X(VarAsSemicolon)                                                            \
  X(VarAsFor)                                                                  \
                                                                               \
  /* Handles `var` after the pattern, either followed by an initializer or the \
   * semicolon.                                                                \
   *                                                                           \
   * If `Equal`:                                                               \
   *   1. Expression                                                           \
   *   2. VarAfterInitializer                                                  \
   * Else:                                                                     \
   *   (state done)                                                            \
   */                                                                          \
  X(VarAfterPattern)                                                           \
                                                                               \
  /* Handles `var` after the initializer, wrapping up its subtree.             \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(VarAfterInitializer)                                                       \
                                                                               \
  /* Handles `var` parsing at the end.                                         \
   *                                                                           \
   * Always:                                                                   \
   *   (state done)                                                            \
   */                                                                          \
  X(VarFinishAsSemicolon)                                                      \
  X(VarFinishAsFor)

CARBON_ENUM_BASE(ParserStateBase, CARBON_PARSER_STATE)

class ParserState : public ParserStateBase<ParserState> {
  using ParserStateBase::ParserStateBase;
};

// We expect ParserState to fit compactly into 8 bits.
static_assert(sizeof(ParserState) == 1, "ParserState includes padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
