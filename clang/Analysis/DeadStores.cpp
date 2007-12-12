//==- DeadStores.cpp - Check for stores to dead variables --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines a DeadStores, a flow-sensitive checker that looks for
//  stores to variables that are no longer live.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/LiveVariables.h"
#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

namespace {
  
class DeadStoreObs : public LiveVariables::ObserverTy {
  ASTContext &Ctx;
  Diagnostic &Diags;
public:
  DeadStoreObs(ASTContext &ctx,Diagnostic &diags) : Ctx(ctx), Diags(diags){}    
  virtual ~DeadStoreObs() {}
  
  virtual void ObserveStmt(Stmt* S,
                           const LiveVariables::AnalysisDataTy& AD,
                           const LiveVariables::ValTy& Live) {
    
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {    
      if (!B->isAssignmentOp()) return; // Skip non-assignments.
      
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS()))
        if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
          if (VD->hasLocalStorage() && !Live(VD,AD)) {
            SourceRange R = B->getRHS()->getSourceRange();
            Diags.Report(Ctx.getFullLoc(DR->getSourceRange().getBegin()),
                         diag::warn_dead_store, 0, 0, &R, 1);                                                                        
        }
    }
    else if(DeclStmt* DS = dyn_cast<DeclStmt>(S))
      // Iterate through the decls.  Warn if any initializers are complex
      // expressions that are not live (never used).
      for (VarDecl* V = cast<VarDecl>(DS->getDecl()); V != NULL ; 
                    V = cast_or_null<VarDecl>(V->getNextDeclarator())) {
        if (V->hasLocalStorage())
          if (Expr* E = V->getInit()) {
            if (!Live(DS->getDecl(),AD)) {
              // Special case: check for initializations with constants.
              //
              //  e.g. : int x = 0;
              //
              // If x is EVER assigned a new value later, don't issue
              // a warning.  This is because such initialization can be
              // due to defensive programming.
              if (!E->isConstantExpr(Ctx,NULL)) {
                // Flag a warning.
                SourceRange R = E->getSourceRange();
                Diags.Report(Ctx.getFullLoc(V->getLocation()),
                             diag::warn_dead_store, 0, 0, &R, 1);
              }
            }
          }
      }
  }
};
  
} // end anonymous namespace

namespace clang {

void CheckDeadStores(CFG& cfg, ASTContext &Ctx, Diagnostic &Diags) {
  LiveVariables L(cfg);
  L.runOnCFG(cfg);
  DeadStoreObs A(Ctx, Diags);
  L.runOnAllBlocks(cfg,A);
}

} // end namespace clang
