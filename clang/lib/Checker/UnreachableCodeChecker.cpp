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
// A similar flow-sensitive only check exists in Analysis/UnreachableCode.cpp
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/ExplodedGraph.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/Checker/BugReporter/BugReporter.h"
#include "GRExprEngineExperimentalChecks.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/SmallPtrSet.h"

// The number of CFGBlock pointers we want to reserve memory for. This is used
// once for each function we analyze.
#define DEFAULT_CFGBLOCKS 256

using namespace clang;

namespace {
class UnreachableCodeChecker : public CheckerVisitor<UnreachableCodeChecker> {
public:
  static void *getTag();
  void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B,
      bool hasWorkRemaining);
private:
  static SourceLocation GetUnreachableLoc(const CFGBlock &b, SourceRange &R);
  void FindUnreachableEntryPoints(const CFGBlock *CB);
  void MarkSuccessorsReachable(const CFGBlock *CB);

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
                                              bool hasWorkRemaining) {
  // Bail out if we didn't cover all paths
  if (hasWorkRemaining)
    return;

  CFG *C = 0;
  // Iterate over ExplodedGraph
  for (ExplodedGraph::node_iterator I = G.nodes_begin(); I != G.nodes_end();
      ++I) {
    const ProgramPoint &P = I->getLocation();

    // Save the CFG if we don't have it already
    if (!C)
      C = P.getLocationContext()->getCFG();

    if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&P)) {
      const CFGBlock *CB = BE->getBlock();
      reachable.insert(CB);
    }
  }

  // Bail out if we didn't get the CFG
  if (!C)
    return;

  // Find CFGBlocks that were not covered by any node
  for (CFG::const_iterator I = C->begin(); I != C->end(); ++I) {
    const CFGBlock *CB = *I;
    // Check if the block is unreachable
    if (!reachable.count(CB)) {
      // Find the entry points for this block
      FindUnreachableEntryPoints(CB);
      // This block may have been pruned; check if we still want to report it
      if (reachable.count(CB))
        continue;

      // We found a statement that wasn't covered
      SourceRange S;
      SourceLocation SL = GetUnreachableLoc(*CB, S);
      if (S.getBegin().isMacroID() || S.getEnd().isMacroID() || S.isInvalid()
          || SL.isInvalid())
        continue;
      B.EmitBasicReport("Unreachable code", "This statement is never executed",
          SL, S);
    }
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
