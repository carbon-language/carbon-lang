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

#include "clang/Checker/LocalCheckers.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/Checker/BugReporter/BugReporter.h"
#include "clang/Checker/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;

namespace {

class DeadStoreObs : public LiveVariables::ObserverTy {
  ASTContext &Ctx;
  BugReporter& BR;
  ParentMap& Parents;
  llvm::SmallPtrSet<VarDecl*, 20> Escaped;

  enum DeadStoreKind { Standard, Enclosing, DeadIncrement, DeadInit };

public:
  DeadStoreObs(ASTContext &ctx, BugReporter& br, ParentMap& parents,
               llvm::SmallPtrSet<VarDecl*, 20> &escaped)
    : Ctx(ctx), BR(br), Parents(parents), Escaped(escaped) {}

  virtual ~DeadStoreObs() {}

  void Report(VarDecl* V, DeadStoreKind dsk, SourceLocation L, SourceRange R) {
    if (Escaped.count(V))
      return;

    std::string name = V->getNameAsString();

    const char* BugType = 0;
    std::string msg;

    switch (dsk) {
      default:
        assert(false && "Impossible dead store type.");

      case DeadInit:
        BugType = "Dead initialization";
        msg = "Value stored to '" + name +
          "' during its initialization is never read";
        break;

      case DeadIncrement:
        BugType = "Dead increment";
      case Standard:
        if (!BugType) BugType = "Dead assignment";
        msg = "Value stored to '" + name + "' is never read";
        break;

      case Enclosing:
        BugType = "Dead nested assignment";
        msg = "Although the value stored to '" + name +
          "' is used in the enclosing expression, the value is never actually"
          " read from '" + name + "'";
        break;
    }

    BR.EmitBasicReport(BugType, "Dead store", msg, L, R);
  }

  void CheckVarDecl(VarDecl* VD, Expr* Ex, Expr* Val,
                    DeadStoreKind dsk,
                    const LiveVariables::AnalysisDataTy& AD,
                    const LiveVariables::ValTy& Live) {

    if (!VD->hasLocalStorage())
      return;
    // Reference types confuse the dead stores checker.  Skip them
    // for now.
    if (VD->getType()->getAs<ReferenceType>())
      return;

    if (!Live(VD, AD) && 
        !(VD->getAttr<UnusedAttr>() || VD->getAttr<BlocksAttr>()))
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
          // Special case: check for assigning null to a pointer.
          //  This is a common form of defensive programming.
          if (VD->getType()->isPointerType()) {
            if (B->getRHS()->isNullPointerConstant(Ctx,
                                              Expr::NPC_ValueDependentIsNull))
              return;
          }

          Expr* RHS = B->getRHS()->IgnoreParenCasts();
          // Special case: self-assignments.  These are often used to shut up
          //  "unused variable" compiler warnings.
          if (DeclRefExpr* RhsDR = dyn_cast<DeclRefExpr>(RHS))
            if (VD == dyn_cast<VarDecl>(RhsDR->getDecl()))
              return;

          // Otherwise, issue a warning.
          DeadStoreKind dsk = Parents.isConsumedExpr(B)
                              ? Enclosing
                              : (isIncrement(VD,B) ? DeadIncrement : Standard);

          CheckVarDecl(VD, DR, B->getRHS(), dsk, AD, Live);
        }
    }
    else if (UnaryOperator* U = dyn_cast<UnaryOperator>(S)) {
      if (!U->isIncrementOp())
        return;

      // Handle: ++x within a subexpression.  The solution is not warn
      //  about preincrements to dead variables when the preincrement occurs
      //  as a subexpression.  This can lead to false negatives, e.g. "(++x);"
      //  A generalized dead code checker should find such issues.
      if (U->isPrefix() && Parents.isConsumedExpr(U))
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
          
        if (V->hasLocalStorage()) {          
          // Reference types confuse the dead stores checker.  Skip them
          // for now.
          if (V->getType()->getAs<ReferenceType>())
            return;
            
          if (Expr* E = V->getInit()) {
            // Don't warn on C++ objects (yet) until we can show that their
            // constructors/destructors don't have side effects.
            if (isa<CXXConstructExpr>(E))
              return;

            if (isa<CXXExprWithTemporaries>(E))
              return;
            
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
              if (E->isConstantInitializer(Ctx))
                return;

              // Special case: check for initializations from constant
              //  variables.
              //
              //  e.g. extern const int MyConstant;
              //       int x = MyConstant;
              //
              if (DeclRefExpr *DRE=dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
                if (VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
                  if (VD->hasGlobalStorage() &&
                      VD->getType().isConstQualified()) return;

              Report(V, DeadInit, V->getLocation(), E->getSourceRange());
            }
          }
        }
      }
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver function to invoke the Dead-Stores checker on a CFG.
//===----------------------------------------------------------------------===//

namespace {
class FindEscaped : public CFGRecStmtDeclVisitor<FindEscaped>{
  CFG *cfg;
public:
  FindEscaped(CFG *c) : cfg(c) {}

  CFG& getCFG() { return *cfg; }

  llvm::SmallPtrSet<VarDecl*, 20> Escaped;

  void VisitUnaryOperator(UnaryOperator* U) {
    // Check for '&'.  Any VarDecl whose value has its address-taken we
    // treat as escaped.
    Expr* E = U->getSubExpr()->IgnoreParenCasts();
    if (U->getOpcode() == UnaryOperator::AddrOf)
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E))
        if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
          Escaped.insert(VD);
          return;
        }
    Visit(E);
  }
};
} // end anonymous namespace


void clang::CheckDeadStores(CFG &cfg, LiveVariables &L, ParentMap &pmap, 
                            BugReporter& BR) {
  FindEscaped FS(&cfg);
  FS.getCFG().VisitBlockStmts(FS);
  DeadStoreObs A(BR.getContext(), BR, pmap, FS.Escaped);
  L.runOnAllBlocks(cfg, &A);
}
