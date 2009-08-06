//===--- SemaTemplateInstantiateExpr.cpp - C++ Template Expr Instantiation ===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation for expressions.
//
//===----------------------------------------------------------------------===/
#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Designator.h"
#include "clang/Lex/Preprocessor.h" // for the identifier table
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN TemplateExprInstantiator 
    : public StmtVisitor<TemplateExprInstantiator, Sema::OwningExprResult> {
    Sema &SemaRef;
    const TemplateArgumentList &TemplateArgs;

  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateExprInstantiator(Sema &SemaRef, 
                             const TemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs) { }

    // Declare VisitXXXStmt nodes for all of the expression kinds.
#define EXPR(Type, Base) OwningExprResult Visit##Type(Type *S);
#define STMT(Type, Base)
#include "clang/AST/StmtNodes.def"

    // Base case. We can't get here.
    Sema::OwningExprResult VisitStmt(Stmt *S) { 
      S->dump();
      assert(false && "Cannot instantiate this kind of expression");
      return SemaRef.ExprError(); 
    }
  };
}

// Base case. We can't get here.
Sema::OwningExprResult TemplateExprInstantiator::VisitExpr(Expr *E) { 
  E->dump();
  assert(false && "Cannot instantiate this kind of expression");
  return SemaRef.ExprError(); 
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitPredefinedExpr(PredefinedExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitIntegerLiteral(IntegerLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitFloatingLiteral(FloatingLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitStringLiteral(StringLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitCharacterLiteral(CharacterLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitGNUNullExpr(GNUNullExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitUnresolvedFunctionNameExpr(
                                              UnresolvedFunctionNameExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitTemplateIdRefExpr(TemplateIdRefExpr *E) {
  TemplateName Template 
    = SemaRef.InstantiateTemplateName(E->getTemplateName(), 
                                      E->getTemplateNameLoc(),
                                      TemplateArgs);
  // FIXME: Can InstantiateTemplateName report an error?
  
  llvm::SmallVector<TemplateArgument, 4> InstantiatedArgs;
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgument InstArg = SemaRef.Instantiate(E->getTemplateArgs()[I],
                                                   TemplateArgs);
    if (InstArg.isNull())
      return SemaRef.ExprError();
    
    InstantiatedArgs.push_back(InstArg);
  }
  
  // FIXME: It's possible that we'll find out now that the template name 
  // actually refers to a type, in which case this is a functional cast. 
  // Implement this!
  
  return SemaRef.BuildTemplateIdExpr(Template, E->getTemplateNameLoc(),
                                     E->getLAngleLoc(),
                                     InstantiatedArgs.data(),
                                     InstantiatedArgs.size(),
                                     E->getRAngleLoc());
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitDeclRefExpr(DeclRefExpr *E) {
  NamedDecl *D = E->getDecl();
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    assert(NTTP->getDepth() == 0 && "No nested templates yet");

    // If the corresponding template argument is NULL or non-existent, it's 
    // because we are performing instantiation from explicitly-specified 
    // template arguments in a function template, but there were some
    // arguments left unspecified.
    if (NTTP->getPosition() >= TemplateArgs.size() ||
        TemplateArgs[NTTP->getPosition()].isNull())
      return SemaRef.Owned(E); // FIXME: Clone the expression!
    
    const TemplateArgument &Arg = TemplateArgs[NTTP->getPosition()]; 
    
    // The template argument itself might be an expression, in which
    // case we just return that expression.
    if (Arg.getKind() == TemplateArgument::Expression)
      // FIXME: Clone the expression!
      return SemaRef.Owned(Arg.getAsExpr());

    if (Arg.getKind() == TemplateArgument::Declaration) {
      ValueDecl *VD = cast<ValueDecl>(Arg.getAsDecl());

      // FIXME: Can VD ever have a dependent type?
      return SemaRef.BuildDeclRefExpr(VD, VD->getType(), E->getLocation(), 
                                      false, false);
    }
    
    assert(Arg.getKind() == TemplateArgument::Integral);
    QualType T = Arg.getIntegralType();
    if (T->isCharType() || T->isWideCharType())
      return SemaRef.Owned(new (SemaRef.Context) CharacterLiteral(
                                          Arg.getAsIntegral()->getZExtValue(),
                                          T->isWideCharType(),
                                          T, 
                                       E->getSourceRange().getBegin()));
    if (T->isBooleanType())
      return SemaRef.Owned(new (SemaRef.Context) CXXBoolLiteralExpr(
                                          Arg.getAsIntegral()->getBoolValue(),
                                                 T, 
                                       E->getSourceRange().getBegin()));

    assert(Arg.getAsIntegral()->getBitWidth() == SemaRef.Context.getIntWidth(T));
    return SemaRef.Owned(new (SemaRef.Context) IntegerLiteral(
                                                 *Arg.getAsIntegral(),
                                                 T, 
                                       E->getSourceRange().getBegin()));
  }

  if (OverloadedFunctionDecl *Ovl = dyn_cast<OverloadedFunctionDecl>(D)) {
    // FIXME: instantiate each decl in the overload set
    return SemaRef.Owned(new (SemaRef.Context) DeclRefExpr(Ovl,
                                                   SemaRef.Context.OverloadTy,
                                                           E->getLocation(),
                                                           false, false));
  }

  NamedDecl *InstD = SemaRef.InstantiateCurrentDeclRef(D);
  if (!InstD)
    return SemaRef.ExprError();

  // FIXME: nested-name-specifier for QualifiedDeclRefExpr
  return SemaRef.BuildDeclarationNameExpr(E->getLocation(), InstD, 
                                          /*FIXME:*/false,
                                          /*FIXME:*/0, 
                                          /*FIXME:*/false);
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
TemplateExprInstantiator::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  Sema::OwningExprResult LHS = Visit(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult RHS = Visit(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  // Since the overloaded array-subscript operator (operator[]) can
  // only be a member function, we can make several simplifying
  // assumptions here:
  //   1) Normal name lookup (from the current scope) will not ever
  //   find any declarations of operator[] that won't also be found be
  //   member operator lookup, so it is safe to pass a NULL Scope
  //   during the instantiation to avoid the lookup entirely.
  //
  //   2) Neither normal name lookup nor argument-dependent lookup at
  //   template definition time will find any operators that won't be
  //   found at template instantiation time, so we do not need to
  //   cache the results of name lookup as we do for the binary
  //   operators.
  SourceLocation LLocFake = ((Expr*)LHS.get())->getSourceRange().getBegin();
  return SemaRef.ActOnArraySubscriptExpr(/*Scope=*/0, move(LHS),
                                         /*FIXME:*/LLocFake,
                                         move(RHS),
                                         E->getRBracketLoc());
}

Sema::OwningExprResult TemplateExprInstantiator::VisitCallExpr(CallExpr *E) {
  // Instantiate callee
  OwningExprResult Callee = Visit(E->getCallee());
  if (Callee.isInvalid())
    return SemaRef.ExprError();

  // Instantiate arguments
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 4> FakeCommaLocs;
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    OwningExprResult Arg = Visit(E->getArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    FakeCommaLocs.push_back(
     SemaRef.PP.getLocForEndOfToken(E->getArg(I)->getSourceRange().getEnd()));
    Args.push_back(Arg.takeAs<Expr>());
  }

  SourceLocation FakeLParenLoc 
    = ((Expr *)Callee.get())->getSourceRange().getBegin();
  return SemaRef.ActOnCallExpr(/*Scope=*/0, move(Callee), 
                               /*FIXME:*/FakeLParenLoc,
                               move_arg(Args),
                               /*FIXME:*/&FakeCommaLocs.front(), 
                               E->getRParenLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitMemberExpr(MemberExpr *E) {
  // Instantiate the base of the expression.
  OwningExprResult Base = Visit(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  // FIXME: Handle declaration names here
  SourceLocation FakeOperatorLoc = 
    SemaRef.PP.getLocForEndOfToken(E->getBase()->getSourceRange().getEnd());
  return SemaRef.ActOnMemberReferenceExpr(/*Scope=*/0,
                                          move(Base), 
                                          /*FIXME*/FakeOperatorLoc,
                                          E->isArrow()? tok::arrow 
                                                      : tok::period,
                                          E->getMemberLoc(),
                               /*FIXME:*/*E->getMemberDecl()->getIdentifier(),
                                   /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  SourceLocation FakeTypeLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
  QualType T = SemaRef.InstantiateType(E->getType(), TemplateArgs,
                                       FakeTypeLoc,
                                       DeclarationName());
  if (T.isNull())
    return SemaRef.ExprError();

  OwningExprResult Init = Visit(E->getInitializer());
  if (Init.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.ActOnCompoundLiteral(E->getLParenLoc(),
                                      T.getAsOpaquePtr(),
                                      /*FIXME*/E->getLParenLoc(),
                                      move(Init));
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
TemplateExprInstantiator::VisitCompoundAssignOperator(
                                                 CompoundAssignOperator *E) {
  return VisitBinaryOperator(E);
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
    OwningExprResult Callee = Visit(E->getCallee());
    if (Callee.isInvalid())
      return SemaRef.ExprError();

    First.release();
    Second.release();

    return SemaRef.Owned(new (SemaRef.Context) CXXOperatorCallExpr(
                                                       SemaRef.Context, 
                                                       E->getOperator(),
                                                       Callee.takeAs<Expr>(), 
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

  // FIXME: This would be far less ugly if CreateOverloadedBinOp took in ExprArg
  // arguments!
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
TemplateExprInstantiator::VisitCXXConditionDeclExpr(CXXConditionDeclExpr *E) {
  VarDecl *Var 
    = cast_or_null<VarDecl>(SemaRef.InstantiateDecl(E->getVarDecl(),
                                                    SemaRef.CurContext,
                                                    TemplateArgs));
  if (!Var)
    return SemaRef.ExprError();

  SemaRef.CurrentInstantiationScope->InstantiatedLocal(E->getVarDecl(), Var);
  return SemaRef.Owned(new (SemaRef.Context) CXXConditionDeclExpr(
                                                    E->getStartLoc(), 
                                                    SourceLocation(),
                                                    Var));
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitConditionalOperator(ConditionalOperator *E) {
  Sema::OwningExprResult Cond = Visit(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult LHS = SemaRef.InstantiateExpr(E->getLHS(), 
                                                       TemplateArgs);
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  Sema::OwningExprResult RHS = Visit(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  if (!E->isTypeDependent()) { 
    // Since our original expression was not type-dependent, we do not
    // perform lookup again at instantiation time (C++ [temp.dep]p1).
    // Instead, we just build the new conditional operator call expression.
    return SemaRef.Owned(new (SemaRef.Context) ConditionalOperator(
                                                           Cond.takeAs<Expr>(),
                                                           LHS.takeAs<Expr>(), 
                                                           RHS.takeAs<Expr>(),
                                                           E->getType()));
  }


  return SemaRef.ActOnConditionalOp(/*FIXME*/E->getCond()->getLocEnd(),
                                    /*FIXME*/E->getFalseExpr()->getLocStart(),
                                    move(Cond), move(LHS), move(RHS));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitAddrLabelExpr(AddrLabelExpr *E) {
  return SemaRef.ActOnAddrLabel(E->getAmpAmpLoc(),
                                E->getLabelLoc(),
                                E->getLabel()->getID());
}

Sema::OwningExprResult TemplateExprInstantiator::VisitStmtExpr(StmtExpr *E) {
  Sema::OwningStmtResult SubStmt 
    = SemaRef.InstantiateCompoundStmt(E->getSubStmt(), TemplateArgs, true);
  if (SubStmt.isInvalid())
    return SemaRef.ExprError();
  
  return SemaRef.ActOnStmtExpr(E->getLParenLoc(), move(SubStmt),
                               E->getRParenLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  assert(false && "__builtin_types_compatible_p is not legal in C++");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  ASTOwningVector<&ActionBase::DeleteExpr> SubExprs(SemaRef);
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I) {
    OwningExprResult SubExpr = Visit(E->getExpr(I));
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();

    SubExprs.push_back(SubExpr.takeAs<Expr>());
  }

  // Find the declaration for __builtin_shufflevector
  const IdentifierInfo &Name 
    = SemaRef.Context.Idents.get("__builtin_shufflevector");
  TranslationUnitDecl *TUDecl = SemaRef.Context.getTranslationUnitDecl();
  DeclContext::lookup_result Lookup = TUDecl->lookup(DeclarationName(&Name));
  assert(Lookup.first != Lookup.second && "No __builtin_shufflevector?");
  
  // Build a reference to the __builtin_shufflevector builtin
  FunctionDecl *Builtin = cast<FunctionDecl>(*Lookup.first);
  Expr *Callee = new (SemaRef.Context) DeclRefExpr(Builtin, Builtin->getType(),
                                                   E->getBuiltinLoc(), 
                                                   false, false);
  SemaRef.UsualUnaryConversions(Callee);

  // Build the CallExpr 
  CallExpr *TheCall = new (SemaRef.Context) CallExpr(SemaRef.Context, Callee,
                                                     SubExprs.takeAs<Expr>(),
                                                     SubExprs.size(),
                                                     Builtin->getResultType(),
                                                     E->getRParenLoc());
  OwningExprResult OwnedCall(SemaRef.Owned(TheCall));

  // Type-check the __builtin_shufflevector expression.
  OwningExprResult Result = SemaRef.SemaBuiltinShuffleVector(TheCall);
  if (Result.isInvalid())
    return SemaRef.ExprError();

  OwnedCall.release();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitChooseExpr(ChooseExpr *E) {
  OwningExprResult Cond = Visit(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult LHS = SemaRef.InstantiateExpr(E->getLHS(), TemplateArgs);
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult RHS = Visit(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.ActOnChooseExpr(E->getBuiltinLoc(),
                                 move(Cond), move(LHS), move(RHS),
                                 E->getRParenLoc());
}

Sema::OwningExprResult TemplateExprInstantiator::VisitVAArgExpr(VAArgExpr *E) {
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  SourceLocation FakeTypeLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getSubExpr()->getSourceRange()
                                       .getEnd());
  QualType T = SemaRef.InstantiateType(E->getType(), TemplateArgs,
                                       /*FIXME:*/FakeTypeLoc, 
                                       DeclarationName());
  if (T.isNull())
    return SemaRef.ExprError();

  return SemaRef.ActOnVAArg(E->getBuiltinLoc(), move(SubExpr),
                            T.getAsOpaquePtr(), E->getRParenLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitInitListExpr(InitListExpr *E) {
  ASTOwningVector<&ActionBase::DeleteExpr, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I) {
    OwningExprResult Init = Visit(E->getInit(I));
    if (Init.isInvalid())
      return SemaRef.ExprError();
    Inits.push_back(Init.takeAs<Expr>());
  }

  return SemaRef.ActOnInitList(E->getLBraceLoc(), move_arg(Inits),
                               E->getRBraceLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  Designation Desig;

  // Instantiate the initializer value
  OwningExprResult Init = Visit(E->getInit());
  if (Init.isInvalid())
    return SemaRef.ExprError();

  // Instantiate the designators.
  ASTOwningVector<&ActionBase::DeleteExpr, 4> ArrayExprs(SemaRef);
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      Desig.AddDesignator(Designator::getField(D->getFieldName(),
                                               D->getDotLoc(),
                                               D->getFieldLoc()));
      continue;
    }

    if (D->isArrayDesignator()) {
      OwningExprResult Index = Visit(E->getArrayIndex(*D));
      if (Index.isInvalid())
        return SemaRef.ExprError();

      Desig.AddDesignator(Designator::getArray(Index.get(), 
                                               D->getLBracketLoc()));

      ArrayExprs.push_back(Index.release());
      continue;
    }

    assert(D->isArrayRangeDesignator() && "New kind of designator?");
    OwningExprResult Start = Visit(E->getArrayRangeStart(*D));
    if (Start.isInvalid())
      return SemaRef.ExprError();

    OwningExprResult End = Visit(E->getArrayRangeEnd(*D));
    if (End.isInvalid())
      return SemaRef.ExprError();

    Desig.AddDesignator(Designator::getArrayRange(Start.get(), 
                                                  End.get(),
                                                  D->getLBracketLoc(),
                                                  D->getEllipsisLoc()));
    
    ArrayExprs.push_back(Start.release());
    ArrayExprs.push_back(End.release());
  }

  OwningExprResult Result = 
    SemaRef.ActOnDesignatedInitializer(Desig,
                                       E->getEqualOrColonLoc(),
                                       E->usesGNUSyntax(),
                                       move(Init));
  if (Result.isInvalid())
    return SemaRef.ExprError();

  ArrayExprs.take();
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitImplicitValueInitExpr(
                                                  ImplicitValueInitExpr *E) {
  assert(!E->isTypeDependent() && !E->isValueDependent() &&
         "ImplicitValueInitExprs are never dependent");
  return SemaRef.Clone(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  OwningExprResult Base = Visit(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  SourceLocation FakeOperatorLoc = 
    SemaRef.PP.getLocForEndOfToken(E->getBase()->getSourceRange().getEnd());
  return SemaRef.ActOnMemberReferenceExpr(/*Scope=*/0,
                                          move(Base), 
                                          /*FIXME*/FakeOperatorLoc,
                                          tok::period,
                                          E->getAccessorLoc(),
                                          E->getAccessor(),
                                   /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitBlockExpr(BlockExpr *E) {
  assert(false && "FIXME:Template instantiation for blocks is unimplemented");
  return SemaRef.ExprError();
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  assert(false && "FIXME:Template instantiation for blocks is unimplemented");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  bool isSizeOf = E->isSizeOf();

  if (E->isArgumentType()) {
    QualType T = E->getArgumentType();
    if (T->isDependentType()) {
      T = SemaRef.InstantiateType(T, TemplateArgs, 
                                  /*FIXME*/E->getOperatorLoc(),
                                &SemaRef.PP.getIdentifierTable().get("sizeof"));
      if (T.isNull())
        return SemaRef.ExprError();
    }

    return SemaRef.CreateSizeOfAlignOfExpr(T, E->getOperatorLoc(), isSizeOf,
                                           E->getSourceRange());
  } 

  Sema::OwningExprResult Arg(SemaRef);
  {   
    // C++0x [expr.sizeof]p1:
    //   The operand is either an expression, which is an unevaluated operand
    //   [...]
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
    
    Arg = Visit(E->getArgumentExpr());
    if (Arg.isInvalid())
      return SemaRef.ExprError();
  }

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
                                             TemplateArgs);
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
                                          E->isAddressOfOperand());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXTemporaryObjectExpr(
                                                  CXXTemporaryObjectExpr *E) {
  QualType T = E->getType();
  if (T->isDependentType()) {
    T = SemaRef.InstantiateType(T, TemplateArgs,
                                E->getTypeBeginLoc(), DeclarationName());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  Args.reserve(E->getNumArgs());
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(), 
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult InstantiatedArg = Visit(*Arg);
    if (InstantiatedArg.isInvalid())
      return SemaRef.ExprError();

    Args.push_back((Expr *)InstantiatedArg.release());
  }

  SourceLocation CommaLoc;
  // FIXME: HACK!
  if (Args.size() > 1) {
    Expr *First = (Expr *)Args[0];
    CommaLoc 
      = SemaRef.PP.getLocForEndOfToken(First->getSourceRange().getEnd());
  }
  return SemaRef.ActOnCXXTypeConstructExpr(SourceRange(E->getTypeBeginLoc()
                                                       /*, FIXME*/),
                                           T.getAsOpaquePtr(),
                                           /*FIXME*/E->getTypeBeginLoc(),
                                           move_arg(Args),
                                           /*HACK*/&CommaLoc,
                                           E->getSourceRange().getEnd());
}

Sema::OwningExprResult TemplateExprInstantiator::VisitCastExpr(CastExpr *E) {
  assert(false && "Cannot instantiate abstract CastExpr");
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
                                           E->getCastKind(),
                                           (Expr *)SubExpr.release(),
                                           E->isLvalueCast());
  return SemaRef.Owned(ICE);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  assert(false && "Cannot instantiate abstract ExplicitCastExpr");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCStyleCastExpr(CStyleCastExpr *E) {
  // Instantiate the type that we're casting to.
  SourceLocation TypeStartLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
  QualType ExplicitTy = SemaRef.InstantiateType(E->getTypeAsWritten(),
                                                TemplateArgs,
                                                TypeStartLoc,
                                                DeclarationName());
  if (ExplicitTy.isNull())
    return SemaRef.ExprError();

  // Instantiate the subexpression.
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  return SemaRef.ActOnCastExpr(E->getLParenLoc(), 
                               ExplicitTy.getAsOpaquePtr(),
                               E->getRParenLoc(),
                               move(SubExpr));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXMemberCallExpr(CXXMemberCallExpr *E) {
  return VisitCallExpr(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXNamedCastExpr(CXXNamedCastExpr *E) {
  // Figure out which cast operator we're dealing with.
  tok::TokenKind Kind;
  switch (E->getStmtClass()) {
  case Stmt::CXXStaticCastExprClass:
    Kind = tok::kw_static_cast;
    break;

  case Stmt::CXXDynamicCastExprClass:
    Kind = tok::kw_dynamic_cast;
    break;

  case Stmt::CXXReinterpretCastExprClass:
    Kind = tok::kw_reinterpret_cast;
    break;

  case Stmt::CXXConstCastExprClass:
    Kind = tok::kw_const_cast;
    break;

  default:
    assert(false && "Invalid C++ named cast");
    return SemaRef.ExprError();
  }

  // Instantiate the type that we're casting to.
  SourceLocation TypeStartLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
  QualType ExplicitTy = SemaRef.InstantiateType(E->getTypeAsWritten(),
                                                TemplateArgs,
                                                TypeStartLoc,
                                                DeclarationName());
  if (ExplicitTy.isNull())
    return SemaRef.ExprError();

  // Instantiate the subexpression.
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  SourceLocation FakeLAngleLoc 
    = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
  SourceLocation FakeRAngleLoc = E->getSubExpr()->getSourceRange().getBegin();
  SourceLocation FakeRParenLoc
    = SemaRef.PP.getLocForEndOfToken(
                                E->getSubExpr()->getSourceRange().getEnd());
  return SemaRef.ActOnCXXNamedCast(E->getOperatorLoc(), Kind,
                                   /*FIXME:*/FakeLAngleLoc,
                                   ExplicitTy.getAsOpaquePtr(),
                                   /*FIXME:*/FakeRAngleLoc,
                                   /*FIXME:*/FakeRAngleLoc,
                                   move(SubExpr),
                                   /*FIXME:*/FakeRParenLoc);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXStaticCastExpr(CXXStaticCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXReinterpretCastExpr(
                                                CXXReinterpretCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXConstCastExpr(CXXConstCastExpr *E) {
  return VisitCXXNamedCastExpr(E);
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitCXXThisExpr(CXXThisExpr *E) {
  QualType ThisType = 
    cast<CXXMethodDecl>(SemaRef.CurContext)->getThisType(SemaRef.Context);
    
  CXXThisExpr *TE = 
    new (SemaRef.Context) CXXThisExpr(E->getLocStart(), ThisType);
  
  return SemaRef.Owned(TE);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  if (E->isTypeOperand()) {
    QualType T = SemaRef.InstantiateType(E->getTypeOperand(),
                                         TemplateArgs,
                                     /*FIXME*/E->getSourceRange().getBegin(),
                                         DeclarationName());
    if (T.isNull())
      return SemaRef.ExprError();

    return SemaRef.ActOnCXXTypeid(E->getSourceRange().getBegin(),
                                  /*FIXME*/E->getSourceRange().getBegin(),
                                  true, T.getAsOpaquePtr(),
                                  E->getSourceRange().getEnd());
  }

  // We don't know whether the expression is potentially evaluated until
  // after we perform semantic analysis, so the expression is potentially
  // potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, 
                                     Action::PotentiallyPotentiallyEvaluated);

  OwningExprResult Operand = Visit(E->getExprOperand());
  if (Operand.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult Result 
    = SemaRef.ActOnCXXTypeid(E->getSourceRange().getBegin(),
                              /*FIXME*/E->getSourceRange().getBegin(),
                             false, Operand.get(),
                             E->getSourceRange().getEnd());
  if (Result.isInvalid())
    return SemaRef.ExprError();

  Operand.release(); // FIXME: since ActOnCXXTypeid silently took ownership
  return move(Result);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXThrowExpr(CXXThrowExpr *E) {
  OwningExprResult SubExpr(SemaRef, (void *)0);
  if (E->getSubExpr()) {
    SubExpr = Visit(E->getSubExpr());
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();
  }

  return SemaRef.ActOnCXXThrow(E->getThrowLoc(), move(SubExpr));
}

Sema::OwningExprResult
TemplateExprInstantiator::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  assert(false && 
         "FIXME: Instantiation for default arguments is unimplemented");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXBindTemporaryExpr(
                                                  CXXBindTemporaryExpr *E) {
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.MaybeBindToTemporary(SubExpr.takeAs<Expr>());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXConstructExpr(CXXConstructExpr *E) {
  assert(!cast<CXXRecordDecl>(E->getConstructor()->getDeclContext())
           ->isDependentType() && "Dependent constructor shouldn't be here");

  QualType T = SemaRef.InstantiateType(E->getType(), TemplateArgs,
                                       /*FIXME*/E->getSourceRange().getBegin(),
                                       DeclarationName());
  if (T.isNull())
    return SemaRef.ExprError();

  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  for (CXXConstructExpr::arg_iterator Arg = E->arg_begin(), 
                                   ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult ArgInst = Visit(*Arg);
    if (ArgInst.isInvalid())
      return SemaRef.ExprError();

    Args.push_back(ArgInst.takeAs<Expr>());
  }

  return SemaRef.Owned(SemaRef.BuildCXXConstructExpr(SemaRef.Context, T,
                                             E->getConstructor(), 
                                             E->isElidable(),
                                             Args.takeAs<Expr>(), 
                                             Args.size()));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXFunctionalCastExpr(
                                                   CXXFunctionalCastExpr *E) {
  // Instantiate the type that we're casting to.
  QualType ExplicitTy = SemaRef.InstantiateType(E->getTypeAsWritten(),
                                                TemplateArgs,
                                                E->getTypeBeginLoc(),
                                                DeclarationName());
  if (ExplicitTy.isNull())
    return SemaRef.ExprError();

  // Instantiate the subexpression.
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();
  
  // FIXME: The end of the type's source range is wrong
  Expr *Sub = SubExpr.takeAs<Expr>();
  return SemaRef.ActOnCXXTypeConstructExpr(SourceRange(E->getTypeBeginLoc()),
                                           ExplicitTy.getAsOpaquePtr(),
                                           /*FIXME:*/E->getTypeBeginLoc(),
                                           Sema::MultiExprArg(SemaRef,
                                                              (void **)&Sub,
                                                              1),
                                           0, 
                                           E->getRParenLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  return SemaRef.Clone(E);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXNewExpr(CXXNewExpr *E) {
  // Instantiate the type that we're allocating
  QualType AllocType = SemaRef.InstantiateType(E->getAllocatedType(),
                                               TemplateArgs,
                                   /*FIXME:*/E->getSourceRange().getBegin(),
                                               DeclarationName());
  if (AllocType.isNull())
    return SemaRef.ExprError();

  // Instantiate the size of the array we're allocating (if any).
  OwningExprResult ArraySize = SemaRef.InstantiateExpr(E->getArraySize(),
                                                       TemplateArgs);
  if (ArraySize.isInvalid())
    return SemaRef.ExprError();

  // Instantiate the placement arguments (if any).
  ASTOwningVector<&ActionBase::DeleteExpr> PlacementArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
    OwningExprResult Arg = Visit(E->getPlacementArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    PlacementArgs.push_back(Arg.take());
  }

  // Instantiate the constructor arguments (if any).
  ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumConstructorArgs(); I != N; ++I) {
    OwningExprResult Arg = Visit(E->getConstructorArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    ConstructorArgs.push_back(Arg.take());
  }

  return SemaRef.BuildCXXNew(E->getSourceRange().getBegin(), 
                             E->isGlobalNew(),
                             /*FIXME*/SourceLocation(),
                             move_arg(PlacementArgs),
                             /*FIXME*/SourceLocation(),
                             E->isParenTypeId(),
                             AllocType,
                             /*FIXME*/E->getSourceRange().getBegin(),
                             SourceRange(),
                             move(ArraySize),
                             /*FIXME*/SourceLocation(),
                             Sema::MultiExprArg(SemaRef,
                                                ConstructorArgs.take(),
                                                ConstructorArgs.size()),
                             E->getSourceRange().getEnd());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  OwningExprResult Operand = Visit(E->getArgument());
  if (Operand.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.ActOnCXXDelete(E->getSourceRange().getBegin(),
                                E->isGlobalDelete(),
                                E->isArrayForm(),
                                move(Operand));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  QualType T = SemaRef.InstantiateType(E->getQueriedType(), TemplateArgs,
                                       /*FIXME*/E->getSourceRange().getBegin(),
                                       DeclarationName());
  if (T.isNull())
    return SemaRef.ExprError();

  SourceLocation FakeLParenLoc
    = SemaRef.PP.getLocForEndOfToken(E->getSourceRange().getBegin());
  return SemaRef.ActOnUnaryTypeTrait(E->getTrait(),
                                     E->getSourceRange().getBegin(),
                                     /*FIXME*/FakeLParenLoc,
                                     T.getAsOpaquePtr(),
                                     E->getSourceRange().getEnd());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitQualifiedDeclRefExpr(QualifiedDeclRefExpr *E) {
  NestedNameSpecifier *NNS 
    = SemaRef.InstantiateNestedNameSpecifier(E->getQualifier(),
                                             E->getQualifierRange(),
                                             TemplateArgs);
  if (!NNS)
    return SemaRef.ExprError();

  CXXScopeSpec SS;
  SS.setRange(E->getQualifierRange());
  SS.setScopeRep(NNS);
  return SemaRef.ActOnDeclarationNameExpr(/*Scope=*/0, 
                                          E->getLocation(),
                                          E->getDecl()->getDeclName(),
                                          /*Trailing lparen=*/false,
                                          &SS,
                                          /*FIXME:*/false);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXExprWithTemporaries(
                                                  CXXExprWithTemporaries *E) {
  OwningExprResult SubExpr = Visit(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  Expr *Temp = 
    SemaRef.MaybeCreateCXXExprWithTemporaries(SubExpr.takeAs<Expr>(),
                                              E->shouldDestroyTemporaries());
  return SemaRef.Owned(Temp);
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXUnresolvedConstructExpr(
                                              CXXUnresolvedConstructExpr *E) {
  QualType T = SemaRef.InstantiateType(E->getTypeAsWritten(), TemplateArgs,
                                       E->getTypeBeginLoc(), 
                                       DeclarationName());
  if (T.isNull())
    return SemaRef.ExprError();

  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 8> FakeCommaLocs;
  for (CXXUnresolvedConstructExpr::arg_iterator Arg = E->arg_begin(),
                                             ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult InstArg = Visit(*Arg);
    if (InstArg.isInvalid())
      return SemaRef.ExprError();

    FakeCommaLocs.push_back(
           SemaRef.PP.getLocForEndOfToken((*Arg)->getSourceRange().getEnd()));
    Args.push_back(InstArg.takeAs<Expr>());
  }

  // FIXME: The end of the type range isn't exactly correct.
  // FIXME: we're faking the locations of the commas
  return SemaRef.ActOnCXXTypeConstructExpr(SourceRange(E->getTypeBeginLoc(),
                                                       E->getLParenLoc()),
                                           T.getAsOpaquePtr(),
                                           E->getLParenLoc(),
                                           move_arg(Args),
                                           &FakeCommaLocs.front(),
                                           E->getRParenLoc());
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitCXXUnresolvedMemberExpr(
                                                 CXXUnresolvedMemberExpr *E) {
  // Instantiate the base of the expression.
  OwningExprResult Base = Visit(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  tok::TokenKind OpKind = E->isArrow() ? tok::arrow : tok::period;
  CXXScopeSpec SS;
  Base = SemaRef.ActOnCXXEnterMemberScope(0, SS, move(Base), OpKind);
  // FIXME: Instantiate the declaration name.
  Base = SemaRef.ActOnMemberReferenceExpr(/*Scope=*/0,
                                          move(Base), E->getOperatorLoc(),
                                          OpKind,
                                          E->getMemberLoc(),
                              /*FIXME:*/*E->getMember().getAsIdentifierInfo(),
                              /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
  SemaRef.ActOnCXXExitMemberScope(0, SS);
  return move(Base);
}

//----------------------------------------------------------------------------
// Objective-C Expressions
//----------------------------------------------------------------------------
Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  return SemaRef.Owned(E->Clone(SemaRef.Context));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  QualType EncodedType = SemaRef.InstantiateType(E->getEncodedType(),
                                                 TemplateArgs,
                                                 /*FIXME:*/E->getAtLoc(),
                                                 DeclarationName());
  if (EncodedType.isNull())
    return SemaRef.ExprError();
  
  return SemaRef.Owned(SemaRef.BuildObjCEncodeExpression(E->getAtLoc(), 
                                                         EncodedType, 
                                                         E->getRParenLoc()));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCMessageExpr(ObjCMessageExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCSelectorExpr(ObjCSelectorExpr *E) { 
  return SemaRef.Owned(E->Clone(SemaRef.Context));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCProtocolExpr(ObjCProtocolExpr *E) { 
  return SemaRef.Owned(E->Clone(SemaRef.Context));
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCKVCRefExpr(ObjCKVCRefExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCSuperExpr(ObjCSuperExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
TemplateExprInstantiator::VisitObjCIsaExpr(ObjCIsaExpr *E) { 
  assert(false && "FIXME: Template instantiations for ObjC expressions");
  return SemaRef.ExprError();
}

Sema::OwningExprResult 
Sema::InstantiateExpr(Expr *E, const TemplateArgumentList &TemplateArgs) {
  if (!E)
    return Owned((Expr *)0);

  TemplateExprInstantiator Instantiator(*this, TemplateArgs);
  return Instantiator.Visit(E);
}
