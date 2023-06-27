// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file defines a number of X-macros that can be used to walk the AST
// class hierarchy. The main entry points are:
//
// -   `CARBON_AST_FOR_EACH_FINAL_CLASS(ACTION)`
//     Invokes `ACTION` on each leaf class in the AST class hierarchy.
// -   `CARBON_AST_FOR_EACH_ABSTRACT_CLASS(ACTION)`
//     Invokes `ACTION` on each non-leaf class in the AST class hierarchy.
// -   `CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(BASE, ACTION)`
//     Invokes `ACTION` on each leaf class in the AST class hierarchy that
//     derives from `BASE`.
// -   `CARBON_AST_FOR_EACH_ABSTRACT_CLASS_BELOW(BASE, ACTION)`
//     Invokes `ACTION` on each non-leaf class in the AST class hierarchy that
//     derives from `BASE`.
//
// These macros are implemented in terms of `CARBON_Foo_KINDS` X-macros. Each of
// these macros takes two macro names as arguments:
//
// -   `CARBON_Foo_KINDS(ABSTRACT, FINAL)`
//     Invokes `ABSTRACT(Class)` for each abstract class that inherits from
//     `Foo`.
//     Invokes `FINAL(Class)` for each final class that inherits from `Foo`.
//
// A fake root class, `AST_RTTI`, is also provided, so that
// `CARBON_AST_RTTI_KINDS` can be used to walk all AST classes.
#ifndef CARBON_EXPLORER_AST_AST_KINDS_H_
#define CARBON_EXPLORER_AST_AST_KINDS_H_

#define CARBON_RTTI_NOOP_ACTION(X)

#define CARBON_AST_FOR_EACH_ABSTRACT_CLASS_BELOW(ROOT, ACTION) \
  CARBON_##ROOT##_KINDS(ACTION, CARBON_RTTI_NOOP_ACTION)
#define CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(ROOT, ACTION) \
  CARBON_##ROOT##_KINDS(CARBON_RTTI_NOOP_ACTION, ACTION)
#define CARBON_AST_FOR_EACH_ABSTRACT_CLASS(ACTION) \
  CARBON_AST_FOR_EACH_ABSTRACT_CLASS_BELOW(AST_RTTI, ACTION)
#define CARBON_AST_FOR_EACH_FINAL_CLASS(ACTION) \
  CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(AST_RTTI, ACTION)

// Class hierarchy description follows, with one _KINDS macro for each abstract
// base class.

#define CARBON_AST_RTTI_KINDS(ABSTRACT, FINAL)            \
  CARBON_AstNode_KINDS(ABSTRACT, FINAL) ABSTRACT(AstNode) \
  CARBON_Element_KINDS(ABSTRACT, FINAL) ABSTRACT(Element)

#define CARBON_AstNode_KINDS(ABSTRACT, FINAL)                     \
  CARBON_Pattern_KINDS(ABSTRACT, FINAL) ABSTRACT(Pattern)         \
  CARBON_Declaration_KINDS(ABSTRACT, FINAL) ABSTRACT(Declaration) \
  FINAL(ImplBinding)                                              \
  FINAL(AlternativeSignature)                                     \
  CARBON_Statement_KINDS(ABSTRACT, FINAL) ABSTRACT(Statement)     \
  CARBON_Expression_KINDS(ABSTRACT, FINAL) ABSTRACT(Expression)   \
  CARBON_WhereClause_KINDS(ABSTRACT, FINAL) ABSTRACT(WhereClause)

#define CARBON_Pattern_KINDS(ABSTRACT, FINAL) \
  FINAL(AutoPattern)                          \
  FINAL(VarPattern)                           \
  FINAL(AddrPattern)                          \
  FINAL(BindingPattern)                       \
  FINAL(GenericBinding)                       \
  FINAL(TuplePattern)                         \
  FINAL(AlternativePattern)                   \
  FINAL(ExpressionPattern)

#define CARBON_Declaration_KINDS(ABSTRACT, FINAL)         \
  FINAL(NamespaceDeclaration)                             \
  CARBON_CallableDeclaration_KINDS(ABSTRACT, FINAL)       \
      ABSTRACT(CallableDeclaration)                       \
  FINAL(SelfDeclaration)                                  \
  FINAL(ClassDeclaration)                                 \
  FINAL(MixinDeclaration)                                 \
  FINAL(MixDeclaration)                                   \
  FINAL(ChoiceDeclaration)                                \
  FINAL(VariableDeclaration)                              \
  CARBON_ConstraintTypeDeclaration_KINDS(ABSTRACT, FINAL) \
      ABSTRACT(ConstraintTypeDeclaration)                 \
  FINAL(InterfaceExtendDeclaration)                       \
  FINAL(InterfaceRequireDeclaration)                      \
  FINAL(AssociatedConstantDeclaration)                    \
  FINAL(ImplDeclaration)                                  \
  FINAL(MatchFirstDeclaration)                            \
  FINAL(AliasDeclaration)                                 \
  FINAL(ExtendBaseDeclaration)

#define CARBON_CallableDeclaration_KINDS(ABSTRACT, FINAL) \
  FINAL(FunctionDeclaration)                              \
  FINAL(DestructorDeclaration)

#define CARBON_ConstraintTypeDeclaration_KINDS(ABSTRACT, FINAL) \
  FINAL(InterfaceDeclaration)                                   \
  FINAL(ConstraintDeclaration)

#define CARBON_Statement_KINDS(ABSTRACT, FINAL)         \
  FINAL(ExpressionStatement)                            \
  FINAL(Assign)                                         \
  FINAL(IncrementDecrement)                             \
  FINAL(VariableDefinition)                             \
  FINAL(If)                                             \
  CARBON_Return_KINDS(ABSTRACT, FINAL) ABSTRACT(Return) \
  FINAL(Block)                                          \
  FINAL(While)                                          \
  FINAL(Break)                                          \
  FINAL(Continue)                                       \
  FINAL(Match)                                          \
  FINAL(For)

#define CARBON_Return_KINDS(ABSTRACT, FINAL) \
  FINAL(ReturnVar)                           \
  FINAL(ReturnExpression)

#define CARBON_Expression_KINDS(ABSTRACT, FINAL)       \
  FINAL(BoolTypeLiteral)                               \
  FINAL(BoolLiteral)                                   \
  FINAL(CallExpression)                                \
  CARBON_ConstantValueLiteral_KINDS(ABSTRACT, FINAL)   \
      ABSTRACT(ConstantValueLiteral)                   \
  CARBON_MemberAccessExpression_KINDS(ABSTRACT, FINAL) \
      ABSTRACT(MemberAccessExpression)                 \
  FINAL(IndexExpression)                               \
  FINAL(IntTypeLiteral)                                \
  FINAL(IntLiteral)                                    \
  FINAL(OperatorExpression)                            \
  FINAL(StringLiteral)                                 \
  FINAL(StringTypeLiteral)                             \
  FINAL(TupleLiteral)                                  \
  FINAL(StructLiteral)                                 \
  FINAL(TypeTypeLiteral)                               \
  FINAL(IdentifierExpression)                          \
  FINAL(DotSelfExpression)                             \
  FINAL(IntrinsicExpression)                           \
  FINAL(IfExpression)                                  \
  FINAL(WhereExpression)                               \
  FINAL(BuiltinConvertExpression)                      \
  FINAL(UnimplementedExpression)

#define CARBON_ConstantValueLiteral_KINDS(ABSTRACT, FINAL) \
  FINAL(FunctionTypeLiteral)                               \
  FINAL(StructTypeLiteral)                                 \
  FINAL(ArrayTypeLiteral)                                  \
  FINAL(ValueLiteral)

#define CARBON_MemberAccessExpression_KINDS(ABSTRACT, FINAL) \
  FINAL(SimpleMemberAccessExpression)                        \
  FINAL(CompoundMemberAccessExpression)                      \
  FINAL(BaseAccessExpression)

#define CARBON_WhereClause_KINDS(ABSTRACT, FINAL) \
  FINAL(ImplsWhereClause)                         \
  FINAL(EqualsWhereClause)                        \
  FINAL(RewriteWhereClause)

#define CARBON_Element_KINDS(ABSTRACT, FINAL) \
  FINAL(NamedElement)                         \
  FINAL(PositionalElement)                    \
  FINAL(BaseElement)

#endif  // CARBON_EXPLORER_AST_AST_KINDS_H_
