//===--- SemaTemplateInstantiateStmt.cpp - C++ Template Stmt Instantiation ===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation for statements.
//
//===----------------------------------------------------------------------===/
#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/Support/Compiler.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN TemplateStmtInstantiator 
    : public StmtVisitor<TemplateStmtInstantiator, Sema::OwningStmtResult> {
    Sema &SemaRef;
    const TemplateArgumentList &TemplateArgs;

  public:
    typedef Sema::OwningExprResult OwningExprResult;
    typedef Sema::OwningStmtResult OwningStmtResult;

    TemplateStmtInstantiator(Sema &SemaRef, 
                             const TemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs) { }

    // FIXME: Once we get closer to completion, replace these
    // manually-written declarations with automatically-generated ones
    // from clang/AST/StmtNodes.def.
    OwningStmtResult VisitDeclStmt(DeclStmt *S);
    OwningStmtResult VisitNullStmt(NullStmt *S);
    OwningStmtResult VisitCompoundStmt(CompoundStmt *S);
    OwningStmtResult VisitIfStmt(IfStmt *S);
    OwningStmtResult VisitExpr(Expr *E);
    OwningStmtResult VisitLabelStmt(LabelStmt *S);
    OwningStmtResult VisitGotoStmt(GotoStmt *S);
    OwningStmtResult VisitReturnStmt(ReturnStmt *S);

    // Base case. I'm supposed to ignore this.
    OwningStmtResult VisitStmt(Stmt *S) { 
      S->dump();
      assert(false && "Cannot instantiate this kind of statement");
      return SemaRef.StmtError(); 
    }
  };
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitDeclStmt(DeclStmt *S) {
  llvm::SmallVector<Decl *, 8> Decls;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    Decl *Instantiated = SemaRef.InstantiateDecl(*D, SemaRef.CurContext, 
                                                 TemplateArgs);
    if (!Instantiated)
      return SemaRef.StmtError();

    Decls.push_back(Instantiated);
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(cast<VarDecl>(*D),
                                                  cast<VarDecl>(Instantiated));
  }

  return SemaRef.Owned(new (SemaRef.Context) DeclStmt(
                                         DeclGroupRef::Create(SemaRef.Context,
                                                              &Decls[0],
                                                              Decls.size()),
                                                      S->getStartLoc(),
                                                      S->getEndLoc()));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitNullStmt(NullStmt *S) {
  return SemaRef.Owned(S->Clone(SemaRef.Context));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitLabelStmt(LabelStmt *S) {
  OwningStmtResult SubStmt = Visit(S->getSubStmt());

  if (SubStmt.isInvalid())
    return SemaRef.StmtError();
  
  // FIXME: Pass the real colon loc in.
  return SemaRef.ActOnLabelStmt(S->getIdentLoc(), S->getID(), SourceLocation(), 
                                move(SubStmt));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitGotoStmt(GotoStmt *S) {
  return SemaRef.ActOnGotoStmt(S->getGotoLoc(), S->getLabelLoc(), 
                               S->getLabel()->getID());
}

Sema::OwningStmtResult
TemplateStmtInstantiator::VisitReturnStmt(ReturnStmt *S) {
  Sema::OwningExprResult Result = SemaRef.ExprEmpty();
  if (Expr *E = S->getRetValue()) {
    Result = SemaRef.InstantiateExpr(E, TemplateArgs);
    
    if (Result.isInvalid())
      return SemaRef.StmtError();
  }
  
  return SemaRef.ActOnReturnStmt(S->getReturnLoc(), move(Result));
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitCompoundStmt(CompoundStmt *S) {
  // FIXME: We need an *easy* RAII way to delete these statements if
  // something goes wrong.
  llvm::SmallVector<Stmt *, 16> Statements;
  
  for (CompoundStmt::body_iterator B = S->body_begin(), BEnd = S->body_end();
       B != BEnd; ++B) {
    OwningStmtResult Result = Visit(*B);
    if (Result.isInvalid()) {
      // FIXME: This should be handled by an RAII destructor.
      for (unsigned I = 0, N = Statements.size(); I != N; ++I)
        Statements[I]->Destroy(SemaRef.Context);
      return SemaRef.StmtError();
    }

    Statements.push_back(Result.takeAs<Stmt>());
  }

  return SemaRef.Owned(
           new (SemaRef.Context) CompoundStmt(SemaRef.Context,
                                              &Statements[0], 
                                              Statements.size(),
                                              S->getLBracLoc(),
                                              S->getRBracLoc()));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitIfStmt(IfStmt *S) {
  // Instantiate the condition
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the "then" branch.
  OwningStmtResult Then = SemaRef.InstantiateStmt(S->getThen(), TemplateArgs);
  if (Then.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the "else" branch.
  OwningStmtResult Else = SemaRef.InstantiateStmt(S->getElse(), TemplateArgs);
  if (Else.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnIfStmt(S->getIfLoc(), move(Cond), move(Then),
                             S->getElseLoc(), move(Else));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitExpr(Expr *E) {
  Sema::OwningExprResult Result = SemaRef.InstantiateExpr(E, TemplateArgs);
  if (Result.isInvalid())
    return SemaRef.StmtError();
  
  return SemaRef.Owned(Result.takeAs<Stmt>());
}

Sema::OwningStmtResult 
Sema::InstantiateStmt(Stmt *S, const TemplateArgumentList &TemplateArgs) {
  if (!S)
    return Owned((Stmt *)0);

  TemplateStmtInstantiator Instantiator(*this, TemplateArgs);
  return Instantiator.Visit(S);
}
