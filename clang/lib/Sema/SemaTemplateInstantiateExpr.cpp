//===--- SemaTemplateInstantiateDecl.cpp - C++ Template Decl Instantiation ===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation for declarations.
//
//===----------------------------------------------------------------------===/
#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Lex/Preprocessor.h" // for the identifier table
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN TemplateExprInstantiator 
    : public StmtVisitor<TemplateExprInstantiator, Sema::OwningExprResult> {
    Sema &SemaRef;
    const TemplateArgument *TemplateArgs;
    unsigned NumTemplateArgs;

  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateExprInstantiator(Sema &SemaRef, 
                             const TemplateArgument *TemplateArgs,
                             unsigned NumTemplateArgs)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs), 
        NumTemplateArgs(NumTemplateArgs) { }

    // FIXME: Once we get closer to completion, replace these
    // manually-written declarations with automatically-generated ones
    // from clang/AST/StmtNodes.def.
    OwningExprResult VisitIntegerLiteral(IntegerLiteral *E);
    OwningExprResult VisitDeclRefExpr(DeclRefExpr *E);
    OwningExprResult VisitParenExpr(ParenExpr *E);
    OwningExprResult VisitUnaryOperator(UnaryOperator *E);
    OwningExprResult VisitBinaryOperator(BinaryOperator *E);
    OwningExprResult VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
    OwningExprResult VisitConditionalOperator(ConditionalOperator *E);
    OwningExprResult VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    OwningExprResult VisitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *E);
    OwningExprResult VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
    OwningExprResult VisitImplicitCastExpr(ImplicitCastExpr *E);
      
    // Base case. I'm supposed to ignore this.
    Sema::OwningExprResult VisitStmt(Stmt *S) { 
      S->dump();
      assert(false && "Cannot instantiate this kind of expression");
      return SemaRef.ExprError(); 
    }
  };
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitIntegerLiteral(IntegerLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitDeclRefExpr(DeclRefExpr *E) {
  Decl *D = E->getDecl();
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    assert(NTTP->getDepth() == 0 && "No nested templates yet");
    const TemplateArgument &Arg = TemplateArgs[NTTP->getPosition()]; 
    QualType T = Arg.getIntegralType();
    if (T->isCharType() || T->isWideCharType())
      return SemaRef.Owned(new (SemaRef.Context) CharacterLiteral(
                                          Arg.getAsIntegral()->getZExtValue(),
                                          T->isWideCharType(),
                                          T, 
                                       E->getSourceRange().getBegin()));
    else if (T->isBooleanType())
      return SemaRef.Owned(new (SemaRef.Context) CXXBoolLiteralExpr(
                                          Arg.getAsIntegral()->getBoolValue(),
                                                 T, 
                                       E->getSourceRange().getBegin()));

    return SemaRef.Owned(new (SemaRef.Context) IntegerLiteral(
                                                 *Arg.getAsIntegral(),
                                                 T, 
                                       E->getSourceRange().getBegin()));
  } else
    assert(false && "Can't handle arbitrary declaration references");

  return SemaRef.ExprError();
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitParenExpr(ParenExpr *E) {
  Sema::OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.Owned(new (SemaRef.Context) ParenExpr(
                                               E->getLParen(), E->getRParen(), 
                                               (Expr *)SubExpr.release()));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitUnaryOperator(UnaryOperator *E) {
  Sema::OwningExprResult Arg = Visit(E->getSubExpr());
  if (Arg.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.CreateBuiltinUnaryOp(E->getOperatorLoc(),
                                      E->getOpcode(),
                                      move(Arg));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitBinaryOperator(BinaryOperator *E) {
  Sema::OwningExprResult LHS = Visit(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult RHS = Visit(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult Result
    = SemaRef.CreateBuiltinBinOp(E->getOperatorLoc(), 
                                 E->getOpcode(),
                                 (Expr *)LHS.get(),
                                 (Expr *)RHS.get());
  if (Result.isInvalid())
    return SemaRef.ExprError();

  LHS.release();
  RHS.release();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  Sema::OwningExprResult First = Visit(E->getArg(0));
  if (First.isInvalid())
    return SemaRef.ExprError();

  Expr *Args[2] = { (Expr *)First.get(), 0 };

  Sema::OwningExprResult Second(SemaRef);
  if (E->getNumArgs() == 2) {
    Second = Visit(E->getArg(1));

    if (Second.isInvalid())
      return SemaRef.ExprError();

    Args[1] = (Expr *)Second.get();
  }

  if (!E->isTypeDependent()) { 
    // Since our original expression was not type-dependent, we do not
    // perform lookup again at instantiation time (C++ [temp.dep]p1).
    // Instead, we just build the new overloaded operator call
    // expression.
    First.release();
    Second.release();
    // FIXME: Don't reuse the callee here. We need to instantiate it.
    return SemaRef.Owned(new (SemaRef.Context) CXXOperatorCallExpr(
                                                       SemaRef.Context, 
                                                       E->getOperator(),
                                                       E->getCallee(), 
                                                       Args, E->getNumArgs(),
                                                       E->getType(), 
                                                       E->getOperatorLoc()));
  }

  bool isPostIncDec = E->getNumArgs() == 2 && 
    (E->getOperator() == OO_PlusPlus || E->getOperator() == OO_MinusMinus);
  if (E->getNumArgs() == 1 || isPostIncDec) {
    if (!Args[0]->getType()->isOverloadableType()) {
      // The argument is not of overloadable type, so try to create a
      // built-in unary operation.
      UnaryOperator::Opcode Opc 
        = UnaryOperator::getOverloadedOpcode(E->getOperator(), isPostIncDec);

      return SemaRef.CreateBuiltinUnaryOp(E->getOperatorLoc(), Opc,
                                          move(First));
    }

    // Fall through to perform overload resolution
  } else {
    assert(E->getNumArgs() == 2 && "Expected binary operation");

    Sema::OwningExprResult Result(SemaRef);
    if (!Args[0]->getType()->isOverloadableType() && 
        !Args[1]->getType()->isOverloadableType()) {
      // Neither of the arguments is an overloadable type, so try to
      // create a built-in binary operation.
      BinaryOperator::Opcode Opc = 
        BinaryOperator::getOverloadedOpcode(E->getOperator());
      Result = SemaRef.CreateBuiltinBinOp(E->getOperatorLoc(), Opc, 
                                          Args[0], Args[1]);
      if (Result.isInvalid())
        return SemaRef.ExprError();

      First.release();
      Second.release();
      return move(Result);
    }

    // Fall through to perform overload resolution.
  }

  // Compute the set of functions that were found at template
  // definition time.
  Sema::FunctionSet Functions;
  DeclRefExpr *DRE = cast<DeclRefExpr>(E->getCallee());
  OverloadedFunctionDecl *Overloads 
    = cast<OverloadedFunctionDecl>(DRE->getDecl());
  
  // FIXME: Do we have to check
  // IsAcceptableNonMemberOperatorCandidate for each of these?
  for (OverloadedFunctionDecl::function_iterator 
         F = Overloads->function_begin(),
         FEnd = Overloads->function_end();
       F != FEnd; ++F)
    Functions.insert(*F);
  
  // Add any functions found via argument-dependent lookup.
  DeclarationName OpName 
    = SemaRef.Context.DeclarationNames.getCXXOperatorName(E->getOperator());
  SemaRef.ArgumentDependentLookup(OpName, Args, E->getNumArgs(), Functions);

  // Create the overloaded operator invocation.
  if (E->getNumArgs() == 1 || isPostIncDec) {
    UnaryOperator::Opcode Opc 
      = UnaryOperator::getOverloadedOpcode(E->getOperator(), isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(E->getOperatorLoc(), Opc,
                                           Functions, move(First));
  }

  // FIXME: This would be far less ugly if CreateOverloadedBinOp took
  // in ExprArg arguments!
  BinaryOperator::Opcode Opc = 
    BinaryOperator::getOverloadedOpcode(E->getOperator());
  OwningExprResult Result 
    = SemaRef.CreateOverloadedBinOp(E->getOperatorLoc(), Opc, 
                                    Functions, Args[0], Args[1]);

  if (Result.isInvalid())
    return SemaRef.ExprError();

  First.release();
  Second.release();
  return move(Result);  
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitConditionalOperator(ConditionalOperator *E) {
  Sema::OwningExprResult Cond = Visit(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();

  // FIXME: use getLHS() and cope with NULLness
  Sema::OwningExprResult True = Visit(E->getTrueExpr());
  if (True.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult False = Visit(E->getFalseExpr());
  if (False.isInvalid())
    return SemaRef.ExprError();

  if (!E->isTypeDependent()) { 
    // Since our original expression was not type-dependent, we do not
    // perform lookup again at instantiation time (C++ [temp.dep]p1).
    // Instead, we just build the new conditional operator call expression.
    return SemaRef.Owned(new (SemaRef.Context) ConditionalOperator(
                                                           Cond.takeAs<Expr>(),
                                                           True.takeAs<Expr>(), 
                                                           False.takeAs<Expr>(),
                                                           E->getType()));
  }


  return SemaRef.ActOnConditionalOp(/*FIXME*/E->getCond()->getLocEnd(),
                                    /*FIXME*/E->getFalseExpr()->getLocStart(),
                                    move(Cond), move(True), move(False));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  bool isSizeOf = E->isSizeOf();

  if (E->isArgumentType()) {
    QualType T = E->getArgumentType();
    if (T->isDependentType()) {
      T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                  /*FIXME*/E->getOperatorLoc(),
                                &SemaRef.PP.getIdentifierTable().get("sizeof"));
      if (T.isNull())
        return SemaRef.ExprError();
    }

    return SemaRef.CreateSizeOfAlignOfExpr(T, E->getOperatorLoc(), isSizeOf,
                                           E->getSourceRange());
  } 

  Sema::OwningExprResult Arg = Visit(E->getArgumentExpr());
  if (Arg.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult Result
    = SemaRef.CreateSizeOfAlignOfExpr((Expr *)Arg.get(), E->getOperatorLoc(),
                                      isSizeOf, E->getSourceRange());
  if (Result.isInvalid())
    return SemaRef.ExprError();

  Arg.release();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *E) {
  NestedNameSpecifier *NNS 
    = SemaRef.InstantiateNestedNameSpecifier(E->getQualifier(), 
                                             E->getQualifierRange(),
                                             TemplateArgs, NumTemplateArgs);
  if (!NNS)
    return SemaRef.ExprError();

  CXXScopeSpec SS;
  SS.setRange(E->getQualifierRange());
  SS.setScopeRep(NNS);

  // FIXME: We're passing in a NULL scope, because
  // ActOnDeclarationNameExpr doesn't actually use the scope when we
  // give it a non-empty scope specifier. Investigate whether it would
  // be better to refactor ActOnDeclarationNameExpr.
  return SemaRef.ActOnDeclarationNameExpr(/*Scope=*/0, E->getLocation(), 
                                          E->getDeclName(), 
                                          /*HasTrailingLParen=*/false,
                                          &SS,
                                          /*FIXME:isAddressOfOperand=*/false);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXTemporaryObjectExpr(
                                                  CXXTemporaryObjectExpr *E) {
  QualType T = E->getType();
  if (T->isDependentType()) {
    T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                E->getTypeBeginLoc(), DeclarationName());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  llvm::SmallVector<Expr *, 16> Args;
  Args.reserve(E->getNumArgs());
  bool Invalid = false;
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(), 
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult InstantiatedArg = Visit(*Arg);
    if (InstantiatedArg.isInvalid()) {
      Invalid = true;
      break;
    }

    Args.push_back((Expr *)InstantiatedArg.release());
  }

  if (!Invalid) {
    SourceLocation CommaLoc;
    // FIXME: HACK!
    if (Args.size() > 1)
      CommaLoc 
        = SemaRef.PP.getLocForEndOfToken(Args[0]->getSourceRange().getEnd());
    Sema::OwningExprResult Result(
      SemaRef.ActOnCXXTypeConstructExpr(SourceRange(E->getTypeBeginLoc()
                                                    /*, FIXME*/),
                                        T.getAsOpaquePtr(),
                                        /*FIXME*/E->getTypeBeginLoc(),
                                        Sema::MultiExprArg(SemaRef,
                                                           (void**)&Args[0],
                                                           Args.size()),
                                        /*HACK*/&CommaLoc,
                                        E->getSourceRange().getEnd()));
    // At this point, Args no longer owns the arguments, no matter what.
    return move(Result);
  }

  // Clean up the instantiated arguments.
  // FIXME: Would rather do this with RAII.
  for (unsigned Idx = 0; Idx < Args.size(); ++Idx)
    SemaRef.DeleteExpr(Args[Idx]);

  return SemaRef.ExprError();
}

Sema::OwningExprResult TemplateExprInstantiator::VisitImplicitCastExpr(
                                                         ImplicitCastExpr *E) {
  assert(!E->isTypeDependent() && "Implicit casts must have known types");

  Sema::OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  ImplicitCastExpr *ICE = 
    new (SemaRef.Context) ImplicitCastExpr(E->getType(),
                                           (Expr *)SubExpr.release(),
                                           E->isLvalueCast());
  return SemaRef.Owned(ICE);
}

Sema::OwningExprResult 
Sema::InstantiateExpr(Expr *E, const TemplateArgument *TemplateArgs,
                      unsigned NumTemplateArgs) {
  TemplateExprInstantiator Instantiator(*this, TemplateArgs, NumTemplateArgs);
  return Instantiator.Visit(E);
}
