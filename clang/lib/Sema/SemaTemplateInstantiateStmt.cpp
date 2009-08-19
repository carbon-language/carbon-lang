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

    template<typename T>
    Sema::FullExprArg FullExpr(T &expr) {
        return SemaRef.FullExpr(expr);
    }
        
  public:
    typedef Sema::OwningExprResult OwningExprResult;
    typedef Sema::OwningStmtResult OwningStmtResult;

    TemplateStmtInstantiator(Sema &SemaRef, 
                             const TemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs) { }

    // Declare VisitXXXStmt nodes for all of the statement kinds.
#define STMT(Type, Base) OwningStmtResult Visit##Type(Type *S);
#define EXPR(Type, Base)
#include "clang/AST/StmtNodes.def"

    // Visit an expression (which will use the expression
    // instantiator).
    OwningStmtResult VisitExpr(Expr *E);

    // Base case. I'm supposed to ignore this.
    OwningStmtResult VisitStmt(Stmt *S) { 
      S->dump();
      assert(false && "Cannot instantiate this kind of statement");
      return SemaRef.StmtError(); 
    }
  };
}

//===----------------------------------------------------------------------===/
//  Common/C statements
//===----------------------------------------------------------------------===/
Sema::OwningStmtResult TemplateStmtInstantiator::VisitDeclStmt(DeclStmt *S) {
  llvm::SmallVector<Decl *, 4> Decls;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    Decl *Instantiated = SemaRef.InstantiateDecl(*D, SemaRef.CurContext, 
                                                 TemplateArgs);
    if (!Instantiated)
      return SemaRef.StmtError();

    Decls.push_back(Instantiated);
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(*D, Instantiated);
  }

  return SemaRef.Owned(new (SemaRef.Context) DeclStmt(
                                         DeclGroupRef::Create(SemaRef.Context,
                                                              &Decls[0],
                                                              Decls.size()),
                                                      S->getStartLoc(),
                                                      S->getEndLoc()));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitNullStmt(NullStmt *S) {
  return SemaRef.Owned(S->Retain());
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
TemplateStmtInstantiator::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  OwningExprResult Target = SemaRef.InstantiateExpr(S->getTarget(),
                                                    TemplateArgs);
  if (Target.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnIndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
                                       move(Target));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitBreakStmt(BreakStmt *S) {
  return SemaRef.Owned(S->Retain());
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitContinueStmt(ContinueStmt *S) {
  return SemaRef.Owned(S->Retain());
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
  return SemaRef.InstantiateCompoundStmt(S, TemplateArgs, false);
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitSwitchCase(SwitchCase *S) {
  assert(false && "SwitchCase statements are never directly instantiated");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitCaseStmt(CaseStmt *S) {
  // The case value expressions are not potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  // Instantiate left-hand case value.
  OwningExprResult LHS = SemaRef.InstantiateExpr(S->getLHS(), TemplateArgs);
  if (LHS.isInvalid())
    return SemaRef.StmtError();

  // Instantiate right-hand case value (for the GNU case-range extension).
  OwningExprResult RHS = SemaRef.InstantiateExpr(S->getRHS(), TemplateArgs);
  if (RHS.isInvalid())
    return SemaRef.StmtError();

  // Build the case statement.
  OwningStmtResult Case = SemaRef.ActOnCaseStmt(S->getCaseLoc(),
                                                move(LHS),
                                                S->getEllipsisLoc(),
                                                move(RHS),
                                                S->getColonLoc());
  if (Case.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the statement following the case
  OwningStmtResult SubStmt = SemaRef.InstantiateStmt(S->getSubStmt(), 
                                                     TemplateArgs);
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();

  SemaRef.ActOnCaseStmtBody(Case.get(), move(SubStmt));
  return move(Case);
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitDefaultStmt(DefaultStmt *S) {
  // Instantiate the statement following the default case
  OwningStmtResult SubStmt = SemaRef.InstantiateStmt(S->getSubStmt(), 
                                                     TemplateArgs);
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnDefaultStmt(S->getDefaultLoc(), 
                                  S->getColonLoc(),
                                  move(SubStmt), 
                                  /*CurScope=*/0);
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitIfStmt(IfStmt *S) {
  // Instantiate the condition
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  Sema::FullExprArg FullCond(FullExpr(Cond));
  
  // Instantiate the "then" branch.
  OwningStmtResult Then = SemaRef.InstantiateStmt(S->getThen(), TemplateArgs);
  if (Then.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the "else" branch.
  OwningStmtResult Else = SemaRef.InstantiateStmt(S->getElse(), TemplateArgs);
  if (Else.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnIfStmt(S->getIfLoc(), FullCond, move(Then),
                             S->getElseLoc(), move(Else));
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitSwitchStmt(SwitchStmt *S) {
  // Instantiate the condition.
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Start the switch statement itself.
  OwningStmtResult Switch = SemaRef.ActOnStartOfSwitchStmt(move(Cond));
  if (Switch.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the body of the switch statement.
  OwningStmtResult Body = SemaRef.InstantiateStmt(S->getBody(), TemplateArgs);
  if (Body.isInvalid())
    return SemaRef.StmtError();

  // Complete the switch statement.
  return SemaRef.ActOnFinishSwitchStmt(S->getSwitchLoc(), move(Switch),
                                       move(Body));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitWhileStmt(WhileStmt *S) {
  // Instantiate the condition
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  Sema::FullExprArg FullCond(FullExpr(Cond));
  
  // Instantiate the body
  OwningStmtResult Body = SemaRef.InstantiateStmt(S->getBody(), TemplateArgs);
  if (Body.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnWhileStmt(S->getWhileLoc(), FullCond, move(Body));
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitDoStmt(DoStmt *S) {
  // Instantiate the condition
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the body
  OwningStmtResult Body = SemaRef.InstantiateStmt(S->getBody(), TemplateArgs);
  if (Body.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnDoStmt(S->getDoLoc(), move(Body), S->getWhileLoc(),
                             SourceLocation(), move(Cond), S->getRParenLoc());
}

Sema::OwningStmtResult TemplateStmtInstantiator::VisitForStmt(ForStmt *S) {
  // Instantiate the initialization statement
  OwningStmtResult Init = SemaRef.InstantiateStmt(S->getInit(), TemplateArgs);
  if (Init.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the condition
  OwningExprResult Cond = SemaRef.InstantiateExpr(S->getCond(), TemplateArgs);
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the increment
  OwningExprResult Inc = SemaRef.InstantiateExpr(S->getInc(), TemplateArgs);
  if (Inc.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the body
  OwningStmtResult Body = SemaRef.InstantiateStmt(S->getBody(), TemplateArgs);
  if (Body.isInvalid())
    return SemaRef.StmtError();

  return SemaRef.ActOnForStmt(S->getForLoc(), S->getLParenLoc(),
                              move(Init), move(Cond), move(Inc),
                              S->getRParenLoc(), move(Body));
}

Sema::OwningStmtResult
TemplateStmtInstantiator::VisitAsmStmt(AsmStmt *S) {
  // FIXME: Implement this 
 assert(false && "Cannot instantiate an 'asm' statement");
  return SemaRef.StmtError();
}

//===----------------------------------------------------------------------===/
//  C++ statements
//===----------------------------------------------------------------------===/
Sema::OwningStmtResult
TemplateStmtInstantiator::VisitCXXTryStmt(CXXTryStmt *S) {
  // Instantiate the try block itself.
  OwningStmtResult TryBlock = VisitCompoundStmt(S->getTryBlock());
  if (TryBlock.isInvalid())
    return SemaRef.StmtError();

  // Instantiate the handlers.
  ASTOwningVector<&ActionBase::DeleteStmt> Handlers(SemaRef);
  for (unsigned I = 0, N = S->getNumHandlers(); I != N; ++I) {
    OwningStmtResult Handler = VisitCXXCatchStmt(S->getHandler(I));
    if (Handler.isInvalid())
      return SemaRef.StmtError();

    Handlers.push_back(Handler.takeAs<Stmt>());
  }

  return SemaRef.ActOnCXXTryBlock(S->getTryLoc(), move(TryBlock),
                                  move_arg(Handlers));
}

Sema::OwningStmtResult
TemplateStmtInstantiator::VisitCXXCatchStmt(CXXCatchStmt *S) {
  // Instantiate the exception declaration, if any.
  VarDecl *Var = 0;
  if (S->getExceptionDecl()) {
    VarDecl *ExceptionDecl = S->getExceptionDecl();
    QualType T = SemaRef.InstantiateType(ExceptionDecl->getType(),
                                         TemplateArgs,
                                         ExceptionDecl->getLocation(),
                                         ExceptionDecl->getDeclName());
    if (T.isNull())
      return SemaRef.StmtError();

    Var = SemaRef.BuildExceptionDeclaration(0, T,
                                            ExceptionDecl->getDeclaratorInfo(),
                                            ExceptionDecl->getIdentifier(),
                                            ExceptionDecl->getLocation(),
                                            /*FIXME: Inaccurate*/
                                   SourceRange(ExceptionDecl->getLocation()));
    if (Var->isInvalidDecl()) {
      Var->Destroy(SemaRef.Context);
      return SemaRef.StmtError();
    }
   
    // Introduce the exception declaration into scope. 
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(ExceptionDecl, Var);
  }

  // Instantiate the actual exception handler.
  OwningStmtResult Handler = Visit(S->getHandlerBlock());
  if (Handler.isInvalid()) {
    if (Var)
      Var->Destroy(SemaRef.Context);
    return SemaRef.StmtError();
  }

  return SemaRef.Owned(new (SemaRef.Context) CXXCatchStmt(S->getCatchLoc(),
                                                          Var,
                                                     Handler.takeAs<Stmt>()));
}

//===----------------------------------------------------------------------===/
//  Objective-C statements
//===----------------------------------------------------------------------===/
Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C @finally statement");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCAtSynchronizedStmt(
                                                ObjCAtSynchronizedStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C @synchronized statement");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCAtTryStmt(ObjCAtTryStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C @try statement");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCForCollectionStmt(
                                               ObjCForCollectionStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C \"for\" statement");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C @throw statement");
  return SemaRef.StmtError();
}

Sema::OwningStmtResult 
TemplateStmtInstantiator::VisitObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate an Objective-C @catch statement");
  return SemaRef.StmtError();
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

Sema::OwningStmtResult 
Sema::InstantiateCompoundStmt(CompoundStmt *S, 
                              const TemplateArgumentList &TemplateArgs,
                              bool isStmtExpr) {
  if (!S)
    return Owned((Stmt *)0);

  TemplateStmtInstantiator Instantiator(*this, TemplateArgs);
  ASTOwningVector<&ActionBase::DeleteStmt> Statements(*this);
  for (CompoundStmt::body_iterator B = S->body_begin(), BEnd = S->body_end();
       B != BEnd; ++B) {
    OwningStmtResult Result = Instantiator.Visit(*B);
    if (Result.isInvalid())
      return StmtError();

    Statements.push_back(Result.takeAs<Stmt>());
  }

  return ActOnCompoundStmt(S->getLBracLoc(), S->getRBracLoc(),
                           move_arg(Statements), isStmtExpr);
}
