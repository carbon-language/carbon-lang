//==- UnreachableCodeChecker.cpp - Generalized dead code checker -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements a generalized unreachable code checker using a
// path-sensitive analysis. We mark any path visited, and then walk the CFG as a
// post-analysis to determine what was never visited.
//
// A similar flow-sensitive only check exists in Analysis/ReachableCode.cpp
//===----------------------------------------------------------------------===//

#include "clang/AST/ParentMap.h"
#include "clang/Basic/Builtins.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/ExplodedGraph.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/Checker/PathSensitive/CheckerHelpers.h"
#include "clang/Checker/BugReporter/BugReporter.h"
#include "GRExprEngineExperimentalChecks.h"
#include "llvm/ADT/SmallPtrSet.h"

// The number of CFGBlock pointers we want to reserve memory for. This is used
// once for each function we analyze.
#define DEFAULT_CFGBLOCKS 256

using namespace clang;

namespace {
class UnreachableCodeChecker : public CheckerVisitor<UnreachableCodeChecker> {
public:
  static void *getTag();
  void VisitEndAnalysis(ExplodedGraph &G,
                        BugReporter &B,
                        GRExprEngine &Eng);
private:
  typedef bool (*ExplodedNodeHeuristic)(const ExplodedNode &EN);

  static SourceLocation GetUnreachableLoc(const CFGBlock &b, SourceRange &R);
  void FindUnreachableEntryPoints(const CFGBlock *CB);
  void MarkSuccessorsReachable(const CFGBlock *CB);
  static const Expr *getConditon(const Stmt *S);
  static bool isInvalidPath(const CFGBlock *CB, const ParentMap &PM);

  llvm::SmallPtrSet<const CFGBlock*, DEFAULT_CFGBLOCKS> reachable;
  llvm::SmallPtrSet<const CFGBlock*, DEFAULT_CFGBLOCKS> visited;
};
}

void *UnreachableCodeChecker::getTag() {
  static int x = 0;
  return &x;
}

void clang::RegisterUnreachableCodeChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UnreachableCodeChecker());
}

void UnreachableCodeChecker::VisitEndAnalysis(ExplodedGraph &G,
                                              BugReporter &B,
                                              GRExprEngine &Eng) {
  // Bail out if we didn't cover all paths
  if (Eng.hasWorkRemaining())
    return;

  CFG *C = 0;
  ParentMap *PM = 0;
  // Iterate over ExplodedGraph
  for (ExplodedGraph::node_iterator I = G.nodes_begin(); I != G.nodes_end();
      ++I) {
    const ProgramPoint &P = I->getLocation();
    const LocationContext *LC = P.getLocationContext();

    // Save the CFG if we don't have it already
    if (!C)
      C = LC->getCFG();
    if (!PM)
      PM = &LC->getParentMap();

    if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&P)) {
      const CFGBlock *CB = BE->getBlock();
      reachable.insert(CB);
    }
  }

  // Bail out if we didn't get the CFG or the ParentMap.
  if (!C || !PM)
    return;

  ASTContext &Ctx = B.getContext();

  // Find CFGBlocks that were not covered by any node
  for (CFG::const_iterator I = C->begin(); I != C->end(); ++I) {
    const CFGBlock *CB = *I;
    // Check if the block is unreachable
    if (reachable.count(CB))
      continue;

    // Find the entry points for this block
    FindUnreachableEntryPoints(CB);

    // This block may have been pruned; check if we still want to report it
    if (reachable.count(CB))
      continue;

    // Check for false positives
    if (CB->size() > 0 && isInvalidPath(CB, *PM))
      continue;

    // We found a statement that wasn't covered
    SourceRange S;
    SourceLocation SL = GetUnreachableLoc(*CB, S);
    if (S.isInvalid() || SL.isInvalid())
      continue;

    // Special case for __builtin_unreachable.
    // FIXME: This should be extended to include other unreachable markers,
    // such as llvm_unreachable.
    if (!CB->empty()) {
      const Stmt *First = CB->front();
      if (const CallExpr *CE = dyn_cast<CallExpr>(First)) {
        if (CE->isBuiltinCall(Ctx) == Builtin::BI__builtin_unreachable)
          continue;
      }
    }

    B.EmitBasicReport("Unreachable code", "This statement is never executed",
        SL, S);
  }
}

// Recursively finds the entry point(s) for this dead CFGBlock.
void UnreachableCodeChecker::FindUnreachableEntryPoints(const CFGBlock *CB) {
  bool allPredecessorsReachable = true;

  visited.insert(CB);

  for (CFGBlock::const_pred_iterator I = CB->pred_begin(); I != CB->pred_end();
      ++I) {
    // Recurse over all unreachable blocks
    if (!reachable.count(*I) && !visited.count(*I)) {
      FindUnreachableEntryPoints(*I);
      allPredecessorsReachable = false;
    }
  }

  // If at least one predecessor is unreachable, mark this block as reachable
  // so we don't report this block.
  if (!allPredecessorsReachable) {
    reachable.insert(CB);
  }
}

// Find the SourceLocation and SourceRange to report this CFGBlock
SourceLocation UnreachableCodeChecker::GetUnreachableLoc(const CFGBlock &b,
                                                         SourceRange &R) {
  const Stmt *S = 0;
  unsigned sn = 0;
  R = SourceRange();

  if (sn < b.size())
    S = b[sn].getStmt();
  else if (b.getTerminator())
    S = b.getTerminator();
  else
    return SourceLocation();

  R = S->getSourceRange();
  return S->getLocStart();
}

// Returns the Expr* condition if this is a conditional statement, or 0
// otherwise
const Expr *UnreachableCodeChecker::getConditon(const Stmt *S) {
  if (const IfStmt *IS = dyn_cast<IfStmt>(S)) {
    return IS->getCond();
  }
  else if (const SwitchStmt *SS = dyn_cast<SwitchStmt>(S)) {
    return SS->getCond();
  }
  else if (const WhileStmt *WS = dyn_cast<WhileStmt>(S)) {
    return WS->getCond();
  }
  else if (const DoStmt *DS = dyn_cast<DoStmt>(S)) {
    return DS->getCond();
  }
  else if (const ForStmt *FS = dyn_cast<ForStmt>(S)) {
    return FS->getCond();
  }
  else if (const ConditionalOperator *CO = dyn_cast<ConditionalOperator>(S)) {
    return CO->getCond();
  }

  return 0;
}

// Traverse the predecessor Stmt*s from this CFGBlock to find any signs of a
// path that is a false positive.
bool UnreachableCodeChecker::isInvalidPath(const CFGBlock *CB,
                                           const ParentMap &PM) {

  // Start at the end of the block and work up the path.
  const Stmt *S = CB->back().getStmt();
  while (S) {
    // Find any false positives in the conditions on this path.
    if (const Expr *cond = getConditon(S)) {
      if (containsMacro(cond) || containsEnum(cond)
          || containsStaticLocal(cond) || containsBuiltinOffsetOf(cond)
          || containsStmt<SizeOfAlignOfExpr>(cond))
        return true;
    }
    // Get the previous statement.
    S = PM.getParent(S);
  }

  return false;
}
