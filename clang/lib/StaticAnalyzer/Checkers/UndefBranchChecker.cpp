//=== UndefBranchChecker.cpp -----------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines UndefBranchChecker, which checks for undefined branch
// condition.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {

class UndefBranchChecker : public Checker<check::BranchCondition> {
  mutable llvm::OwningPtr<BuiltinBug> BT;

  struct FindUndefExpr {
    ProgramStateManager& VM;
    const ProgramState *St;

    FindUndefExpr(ProgramStateManager& V, const ProgramState *S) : VM(V), St(S) {}

    const Expr *FindExpr(const Expr *Ex) {
      if (!MatchesCriteria(Ex))
        return 0;

      for (Stmt::const_child_iterator I = Ex->child_begin(), 
                                      E = Ex->child_end();I!=E;++I)
        if (const Expr *ExI = dyn_cast_or_null<Expr>(*I)) {
          const Expr *E2 = FindExpr(ExI);
          if (E2) return E2;
        }

      return Ex;
    }

    bool MatchesCriteria(const Expr *Ex) { return St->getSVal(Ex).isUndef(); }
  };

public:
  void checkBranchCondition(const Stmt *Condition, BranchNodeBuilder &Builder,
                            ExprEngine &Eng) const;
};

}

void UndefBranchChecker::checkBranchCondition(const Stmt *Condition,
                                              BranchNodeBuilder &Builder,
                                              ExprEngine &Eng) const {
  const ProgramState *state = Builder.getState();
  SVal X = state->getSVal(Condition);
  if (X.isUndef()) {
    ExplodedNode *N = Builder.generateNode(state, true);
    if (N) {
      N->markAsSink();
      if (!BT)
        BT.reset(
               new BuiltinBug("Branch condition evaluates to a garbage value"));

      // What's going on here: we want to highlight the subexpression of the
      // condition that is the most likely source of the "uninitialized
      // branch condition."  We do a recursive walk of the condition's
      // subexpressions and roughly look for the most nested subexpression
      // that binds to Undefined.  We then highlight that expression's range.
      BlockEdge B = cast<BlockEdge>(N->getLocation());
      const Expr *Ex = cast<Expr>(B.getSrc()->getTerminatorCondition());
      assert (Ex && "Block must have a terminator.");

      // Get the predecessor node and check if is a PostStmt with the Stmt
      // being the terminator condition.  We want to inspect the state
      // of that node instead because it will contain main information about
      // the subexpressions.
      assert (!N->pred_empty());

      // Note: any predecessor will do.  They should have identical state,
      // since all the BlockEdge did was act as an error sink since the value
      // had to already be undefined.
      ExplodedNode *PrevN = *N->pred_begin();
      ProgramPoint P = PrevN->getLocation();
      const ProgramState *St = N->getState();

      if (PostStmt *PS = dyn_cast<PostStmt>(&P))
        if (PS->getStmt() == Ex)
          St = PrevN->getState();

      FindUndefExpr FindIt(Eng.getStateManager(), St);
      Ex = FindIt.FindExpr(Ex);

      // Emit the bug report.
      EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getDescription(),N);
      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Ex);
      R->addRange(Ex->getSourceRange());

      Eng.getBugReporter().EmitReport(R);
    }

    Builder.markInfeasible(true);
    Builder.markInfeasible(false);
  }
}

void ento::registerUndefBranchChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefBranchChecker>();
}
