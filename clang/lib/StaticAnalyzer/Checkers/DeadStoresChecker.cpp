//==- DeadStoresChecker.cpp - Check for stores to dead variables -*- C++ -*-==//
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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;
using namespace ento;

namespace {

// FIXME: Eventually migrate into its own file, and have it managed by
// AnalysisManager.
class ReachableCode {
  const CFG &cfg;
  llvm::BitVector reachable;
public:
  ReachableCode(const CFG &cfg)
    : cfg(cfg), reachable(cfg.getNumBlockIDs(), false) {}
  
  void computeReachableBlocks();
  
  bool isReachable(const CFGBlock *block) const {
    return reachable[block->getBlockID()];
  }
};
}

void ReachableCode::computeReachableBlocks() {
  if (!cfg.getNumBlockIDs())
    return;
  
  SmallVector<const CFGBlock*, 10> worklist;
  worklist.push_back(&cfg.getEntry());
  
  while (!worklist.empty()) {
    const CFGBlock *block = worklist.back();
    worklist.pop_back();
    llvm::BitVector::reference isReachable = reachable[block->getBlockID()];
    if (isReachable)
      continue;
    isReachable = true;
    for (CFGBlock::const_succ_iterator i = block->succ_begin(),
                                       e = block->succ_end(); i != e; ++i)
      if (const CFGBlock *succ = *i)
        worklist.push_back(succ);
  }
}

namespace {
class DeadStoreObs : public LiveVariables::Observer {
  const CFG &cfg;
  ASTContext &Ctx;
  BugReporter& BR;
  ParentMap& Parents;
  llvm::SmallPtrSet<const VarDecl*, 20> Escaped;
  llvm::OwningPtr<ReachableCode> reachableCode;
  const CFGBlock *currentBlock;

  enum DeadStoreKind { Standard, Enclosing, DeadIncrement, DeadInit };

public:
  DeadStoreObs(const CFG &cfg, ASTContext &ctx,
               BugReporter& br, ParentMap& parents,
               llvm::SmallPtrSet<const VarDecl*, 20> &escaped)
    : cfg(cfg), Ctx(ctx), BR(br), Parents(parents),
      Escaped(escaped), currentBlock(0) {}

  virtual ~DeadStoreObs() {}

  void Report(const VarDecl* V, DeadStoreKind dsk,
              SourceLocation L, SourceRange R) {
    if (Escaped.count(V))
      return;
    
    // Compute reachable blocks within the CFG for trivial cases
    // where a bogus dead store can be reported because itself is unreachable.
    if (!reachableCode.get()) {
      reachableCode.reset(new ReachableCode(cfg));
      reachableCode->computeReachableBlocks();
    }
    
    if (!reachableCode->isReachable(currentBlock))
      return;

    const std::string &name = V->getNameAsString();

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
        // Don't report issues in this case, e.g.: "if (x = foo())",
        // where 'x' is unused later.  We have yet to see a case where 
        // this is a real bug.
        return;
    }

    BR.EmitBasicReport(BugType, "Dead store", msg, L, R);
  }

  void CheckVarDecl(const VarDecl* VD, const Expr* Ex, const Expr* Val,
                    DeadStoreKind dsk,
                    const LiveVariables::LivenessValues &Live) {

    if (!VD->hasLocalStorage())
      return;
    // Reference types confuse the dead stores checker.  Skip them
    // for now.
    if (VD->getType()->getAs<ReferenceType>())
      return;

    if (!Live.isLive(VD) && 
        !(VD->getAttr<UnusedAttr>() || VD->getAttr<BlocksAttr>()))
      Report(VD, dsk, Ex->getSourceRange().getBegin(),
             Val->getSourceRange());
  }

  void CheckDeclRef(const DeclRefExpr* DR, const Expr* Val, DeadStoreKind dsk,
                    const LiveVariables::LivenessValues& Live) {
    if (const VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl()))
      CheckVarDecl(VD, DR, Val, dsk, Live);
  }

  bool isIncrement(VarDecl* VD, const BinaryOperator* B) {
    if (B->isCompoundAssignmentOp())
      return true;

    const Expr* RHS = B->getRHS()->IgnoreParenCasts();
    const BinaryOperator* BRHS = dyn_cast<BinaryOperator>(RHS);

    if (!BRHS)
      return false;

    const DeclRefExpr *DR;

    if ((DR = dyn_cast<DeclRefExpr>(BRHS->getLHS()->IgnoreParenCasts())))
      if (DR->getDecl() == VD)
        return true;

    if ((DR = dyn_cast<DeclRefExpr>(BRHS->getRHS()->IgnoreParenCasts())))
      if (DR->getDecl() == VD)
        return true;

    return false;
  }

  virtual void observeStmt(const Stmt* S, const CFGBlock *block,
                           const LiveVariables::LivenessValues &Live) {

    currentBlock = block;
    
    // Skip statements in macros.
    if (S->getLocStart().isMacroID())
      return;

    // Only cover dead stores from regular assignments.  ++/-- dead stores
    // have never flagged a real bug.
    if (const BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {
      if (!B->isAssignmentOp()) return; // Skip non-assignments.

      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS()))
        if (VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
          // Special case: check for assigning null to a pointer.
          //  This is a common form of defensive programming.
          QualType T = VD->getType();
          if (T->isPointerType() || T->isObjCObjectPointerType()) {
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

          CheckVarDecl(VD, DR, B->getRHS(), dsk, Live);
        }
    }
    else if (const UnaryOperator* U = dyn_cast<UnaryOperator>(S)) {
      if (!U->isIncrementOp() || U->isPrefix())
        return;

      const Stmt *parent = Parents.getParentIgnoreParenCasts(U);
      if (!parent || !isa<ReturnStmt>(parent))
        return;

      const Expr *Ex = U->getSubExpr()->IgnoreParenCasts();

      if (const DeclRefExpr* DR = dyn_cast<DeclRefExpr>(Ex))
        CheckDeclRef(DR, U, DeadIncrement, Live);
    }
    else if (const DeclStmt* DS = dyn_cast<DeclStmt>(S))
      // Iterate through the decls.  Warn if any initializers are complex
      // expressions that are not live (never used).
      for (DeclStmt::const_decl_iterator DI=DS->decl_begin(), DE=DS->decl_end();
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
            while (ExprWithCleanups *exprClean = dyn_cast<ExprWithCleanups>(E))
              E = exprClean->getSubExpr();
            
            // Don't warn on C++ objects (yet) until we can show that their
            // constructors/destructors don't have side effects.
            if (isa<CXXConstructExpr>(E))
              return;
            
            // A dead initialization is a variable that is dead after it
            // is initialized.  We don't flag warnings for those variables
            // marked 'unused'.
            if (!Live.isLive(V) && V->getAttr<UnusedAttr>() == 0) {
              // Special case: check for initializations with constants.
              //
              //  e.g. : int x = 0;
              //
              // If x is EVER assigned a new value later, don't issue
              // a warning.  This is because such initialization can be
              // due to defensive programming.
              if (E->isConstantInitializer(Ctx, false))
                return;

              if (DeclRefExpr *DRE=dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
                if (VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
                  // Special case: check for initialization from constant
                  //  variables.
                  //
                  //  e.g. extern const int MyConstant;
                  //       int x = MyConstant;
                  //
                  if (VD->hasGlobalStorage() &&
                      VD->getType().isConstQualified())
                    return;
                  // Special case: check for initialization from scalar
                  //  parameters.  This is often a form of defensive
                  //  programming.  Non-scalars are still an error since
                  //  because it more likely represents an actual algorithmic
                  //  bug.
                  if (isa<ParmVarDecl>(VD) && VD->getType()->isScalarType())
                    return;
                }

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

  llvm::SmallPtrSet<const VarDecl*, 20> Escaped;

  void VisitUnaryOperator(UnaryOperator* U) {
    // Check for '&'.  Any VarDecl whose value has its address-taken we
    // treat as escaped.
    Expr* E = U->getSubExpr()->IgnoreParenCasts();
    if (U->getOpcode() == UO_AddrOf)
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E))
        if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
          Escaped.insert(VD);
          return;
        }
    Visit(E);
  }
};
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// DeadStoresChecker
//===----------------------------------------------------------------------===//

namespace {
class DeadStoresChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    if (LiveVariables *L = mgr.getLiveVariables(D)) {
      CFG &cfg = *mgr.getCFG(D);
      ParentMap &pmap = mgr.getParentMap(D);
      FindEscaped FS(&cfg);
      FS.getCFG().VisitBlockStmts(FS);
      DeadStoreObs A(cfg, BR.getContext(), BR, pmap, FS.Escaped);
      L->runOnAllBlocks(A);
    }
  }
};
}

void ento::registerDeadStoresChecker(CheckerManager &mgr) {
  mgr.registerChecker<DeadStoresChecker>();
}
