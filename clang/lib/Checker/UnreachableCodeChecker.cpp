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
#include "clang/Basic/SourceManager.h"
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
  static inline const Stmt *getUnreachableStmt(const CFGBlock *CB);
  void FindUnreachableEntryPoints(const CFGBlock *CB);
  static bool isInvalidPath(const CFGBlock *CB, const ParentMap &PM);

  llvm::SmallSet<unsigned, DEFAULT_CFGBLOCKS> reachable;
  llvm::SmallSet<unsigned, DEFAULT_CFGBLOCKS> visited;
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
      C = LC->getAnalysisContext()->getUnoptimizedCFG();
    if (!PM)
      PM = &LC->getParentMap();

    if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&P)) {
      const CFGBlock *CB = BE->getBlock();
      reachable.insert(CB->getBlockID());
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
    if (reachable.count(CB->getBlockID()))
      continue;

    // Find the entry points for this block
    FindUnreachableEntryPoints(CB);

    // This block may have been pruned; check if we still want to report it
    if (reachable.count(CB->getBlockID()))
      continue;

    // Check for false positives
    if (CB->size() > 0 && isInvalidPath(CB, *PM))
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

    // We found a block that wasn't covered - find the statement to report
    SourceRange SR;
    SourceLocation SL;
    if (const Stmt *S = getUnreachableStmt(CB)) {
      SR = S->getSourceRange();
      SL = S->getLocStart();
      if (SR.isInvalid() || SL.isInvalid())
        continue;
    }
    else
      continue;

    // Check if the SourceLocation is in a system header
    const SourceManager &SM = B.getSourceManager();
    if (SM.isInSystemHeader(SL) || SM.isInExternCSystemHeader(SL))
      continue;

    B.EmitBasicReport("Unreachable code", "Dead code", "This statement is never"
        " executed", SL, SR);
  }
}

// Recursively finds the entry point(s) for this dead CFGBlock.
void UnreachableCodeChecker::FindUnreachableEntryPoints(const CFGBlock *CB) {
  bool allPredecessorsReachable = true;

  visited.insert(CB->getBlockID());

  for (CFGBlock::const_pred_iterator I = CB->pred_begin(); I != CB->pred_end();
      ++I) {
    // Recurse over all unreachable blocks
    if (!reachable.count((*I)->getBlockID())) {
      // At least one predeccessor was unreachable
      allPredecessorsReachable = false;

      // Only visit the block once
      if (!visited.count((*I)->getBlockID()))
        FindUnreachableEntryPoints(*I);
    }
  }

  // If at least one predecessor is unreachable, mark this block as reachable
  // so we don't report this block.
  if (!allPredecessorsReachable) {
    reachable.insert(CB->getBlockID());
  }
}

// Find the Stmt* in a CFGBlock for reporting a warning
const Stmt *UnreachableCodeChecker::getUnreachableStmt(const CFGBlock *CB) {
  if (CB->size() > 0)
    return CB->front().getStmt();
  else if (const Stmt *S = CB->getTerminator())
    return S;
  else
    return 0;
}

// Determines if the path to this CFGBlock contained an element that infers this
// block is a false positive. We assume that FindUnreachableEntryPoints has
// already marked only the entry points to any dead code, so we need only to
// find the condition that led to this block (the predecessor of this block.)
// There will never be more than one predecessor.
bool UnreachableCodeChecker::isInvalidPath(const CFGBlock *CB,
                                           const ParentMap &PM) {
  // Assert this CFGBlock only has one or zero predecessors
  assert(CB->pred_size() == 0 || CB->pred_size() == 1);

  // If there are no predecessors, then this block is trivially unreachable
  if (CB->pred_size() == 0)
    return false;

  const CFGBlock *pred = *CB->pred_begin();

  // Get the predecessor block's terminator conditon
  const Stmt *cond = pred->getTerminatorCondition();
  assert(cond && "CFGBlock's predecessor has a terminator condition");

  // Run each of the checks on the conditions
  if (containsMacro(cond) || containsEnum(cond)
      || containsStaticLocal(cond) || containsBuiltinOffsetOf(cond)
      || containsStmt<SizeOfAlignOfExpr>(cond))
    return true;

  return false;
}
