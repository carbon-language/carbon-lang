//==- DeadStores.cpp - Check for stores to dead variables --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DeadStores, a flow-sensitive checker that looks for
//  stores to variables that are no longer live.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMap.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {

class VISIBILITY_HIDDEN DeadStoreObs : public LiveVariables::ObserverTy {
  ASTContext &Ctx;
  BugReporter& BR;
  ParentMap& Parents;
  
  enum DeadStoreKind { Standard, Enclosing, DeadIncrement, DeadInit };
    
public:
  DeadStoreObs(ASTContext &ctx, BugReporter& br, ParentMap& parents)
    : Ctx(ctx), BR(br), Parents(parents) {}
  
  virtual ~DeadStoreObs() {}
  
  void Report(VarDecl* V, DeadStoreKind dsk, SourceLocation L, SourceRange R) {

    std::string name(V->getName());
    
    const char* BugType = 0;
    std::string msg;
    
    switch (dsk) {
      default:
        assert(false && "Impossible dead store type.");
        
      case DeadInit:
        BugType = "dead initialization";
        msg = "Value stored to '" + name +
          "' during its initialization is never read";
        break;
        
      case DeadIncrement:
        BugType = "dead increment";
      case Standard:
        if (!BugType) BugType = "dead store";
        msg = "Value stored to '" + name + "' is never read";
        break;
        
      case Enclosing:
        BugType = "dead store";
        msg = "Although the value stored to '" + name +
          "' is used in the enclosing expression, the value is never actually"
          " read from '" + name + "'";
        break;
    }
      
    BR.EmitBasicReport(BugType, msg.c_str(), L, R);      
  }
  
  void CheckVarDecl(VarDecl* VD, Expr* Ex, Expr* Val,
                    DeadStoreKind dsk,
                    const LiveVariables::AnalysisDataTy& AD,
                    const LiveVariables::ValTy& Live) {

    if (VD->hasLocalStorage() && !Live(VD, AD) && !VD->getAttr<UnusedAttr>())
      Report(VD, dsk, Ex->getSourceRange().getBegin(),
             Val->getSourceRange());      
  }
  
  void CheckDeclRef(DeclRefExpr* DR, Expr* Val, DeadStoreKind dsk,
                    const LiveVariables::AnalysisDataTy& AD,
                    const LiveVariables::ValTy& Live) {
    
    if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
      CheckVarDecl(VD, DR, Val, dsk, AD, Live);
  }
  
  bool isIncrement(VarDecl* VD, BinaryOperator* B) {
    if (B->isCompoundAssignmentOp())
      return true;
    
    Expr* RHS = B->getRHS()->IgnoreParenCasts();
    BinaryOperator* BRHS = dyn_cast<BinaryOperator>(RHS);
    
    if (!BRHS)
      return false;
    
    DeclRefExpr *DR;
    
    if ((DR = dyn_cast<DeclRefExpr>(BRHS->getLHS()->IgnoreParenCasts())))
      if (DR->getDecl() == VD)
        return true;
    
    if ((DR = dyn_cast<DeclRefExpr>(BRHS->getRHS()->IgnoreParenCasts())))
      if (DR->getDecl() == VD)
        return true;
    
    return false;
  }
  
  virtual void ObserveStmt(Stmt* S,
                           const LiveVariables::AnalysisDataTy& AD,
                           const LiveVariables::ValTy& Live) {
    
    // Skip statements in macros.
    if (S->getLocStart().isMacroID())
      return;
    
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {    
      if (!B->isAssignmentOp()) return; // Skip non-assignments.
      
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS()))
        if (VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
     
          // Special case: check for assigning null to a pointer.  This
          //  is a common form of defensive programming.
          // FIXME: Make this optional?
          
          Expr* Val = B->getRHS();
          llvm::APSInt Result(Ctx.getTypeSize(Val->getType()));
          
          if (VD->getType()->isPointerType() &&
              Val->IgnoreParenCasts()->isIntegerConstantExpr(Result, Ctx, 0))
            if (Result == 0)
              return;

          DeadStoreKind dsk = 
            Parents.isSubExpr(B)
            ? Enclosing 
            : (isIncrement(VD,B) ? DeadIncrement : Standard);
          
          CheckVarDecl(VD, DR, Val, dsk, AD, Live);
        }              
    }
    else if (UnaryOperator* U = dyn_cast<UnaryOperator>(S)) {
      if (!U->isIncrementOp())
        return;

      Expr *Ex = U->getSubExpr()->IgnoreParenCasts();
      
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(Ex))
        CheckDeclRef(DR, U, DeadIncrement, AD, Live);
    }    
    else if (DeclStmt* DS = dyn_cast<DeclStmt>(S))
      // Iterate through the decls.  Warn if any initializers are complex
      // expressions that are not live (never used).
      for (DeclStmt::decl_iterator DI=DS->decl_begin(), DE=DS->decl_end();
           DI != DE; ++DI) {
        
        VarDecl* V = dyn_cast<VarDecl>(*DI);

        if (!V)
          continue;
        
        if (V->hasLocalStorage())
          if (Expr* E = V->getInit()) {
            // A dead initialization is a variable that is dead after it
            // is initialized.  We don't flag warnings for those variables
            // marked 'unused'.
            if (!Live(V, AD) && V->getAttr<UnusedAttr>() == 0) {
              // Special case: check for initializations with constants.
              //
              //  e.g. : int x = 0;
              //
              // If x is EVER assigned a new value later, don't issue
              // a warning.  This is because such initialization can be
              // due to defensive programming.
              if (!E->isConstantExpr(Ctx,NULL))
                Report(V, DeadInit, V->getLocation(), E->getSourceRange());
            }
          }
      }
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver function to invoke the Dead-Stores checker on a CFG.
//===----------------------------------------------------------------------===//

void clang::CheckDeadStores(LiveVariables& L, BugReporter& BR) {  
  DeadStoreObs A(BR.getContext(), BR, BR.getParentMap());
  L.runOnAllBlocks(*BR.getCFG(), &A);
}
