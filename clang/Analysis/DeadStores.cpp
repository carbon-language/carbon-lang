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

#include "clang/AST/Expr.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/LiveVariables.h"
#include "clang/AST/CFG.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;

namespace {

class DeadStoreAuditor : public LiveVariablesAuditor {
  Preprocessor& PP;
public:
  DeadStoreAuditor(Preprocessor& pp) : PP(pp) {}
  virtual ~DeadStoreAuditor() {}

  virtual void AuditStmt(Stmt* S, LiveVariables& L, llvm::BitVector& Live) {                                 
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {    
      // Is this an assignment?
      if (!B->isAssignmentOp())
        return;
      
      // Is this an assignment to a variable?
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS())) {
        // Is the variable live?
        if (!L.isLive(Live,DR->getDecl())) {
          SourceRange R = B->getRHS()->getSourceRange();
          PP.getDiagnostics().Report(DR->getSourceRange().Begin(),
                                     diag::warn_dead_store, 0, 0,
                                     &R,1);
                                                                        
        }
      }
    }
  }
};
  
} // end anonymous namespace

namespace clang {

void CheckDeadStores(CFG& cfg, LiveVariables& L, Preprocessor& PP) {
  DeadStoreAuditor A(PP);
  
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I)
    L.runOnBlock(&(*I),&A);
}

void CheckDeadStores(CFG& cfg, Preprocessor& PP) {
  LiveVariables L;
  L.runOnCFG(cfg);
  CheckDeadStores(cfg,L,PP);
}

} // end namespace clang