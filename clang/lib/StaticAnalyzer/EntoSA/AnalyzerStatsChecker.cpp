//==--AnalyzerStatsChecker.cpp - Analyzer visitation statistics --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file reports various statistics about analyzer visitation.
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/BugReporter/BugReporter.h"

// FIXME: Restructure checker registration.
#include "Checkers/ExprEngineExperimentalChecks.h"

#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;
using namespace ento;

namespace {
class AnalyzerStatsChecker : public CheckerVisitor<AnalyzerStatsChecker> {
public:
  static void *getTag();
  void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B, ExprEngine &Eng);

private:
  llvm::SmallPtrSet<const CFGBlock*, 256> reachable;
};
}

void *AnalyzerStatsChecker::getTag() {
  static int x = 0;
  return &x;
}

void ento::RegisterAnalyzerStatsChecker(ExprEngine &Eng) {
  Eng.registerCheck(new AnalyzerStatsChecker());
}

void AnalyzerStatsChecker::VisitEndAnalysis(ExplodedGraph &G,
                                            BugReporter &B,
                                            ExprEngine &Eng) {
  const CFG *C  = 0;
  const Decl *D = 0;
  const LocationContext *LC = 0;
  const SourceManager &SM = B.getSourceManager();

  // Iterate over explodedgraph
  for (ExplodedGraph::node_iterator I = G.nodes_begin();
      I != G.nodes_end(); ++I) {
    const ProgramPoint &P = I->getLocation();
    // Save the LocationContext if we don't have it already
    if (!LC)
      LC = P.getLocationContext();

    if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&P)) {
      const CFGBlock *CB = BE->getBlock();
      reachable.insert(CB);
    }
  }

  // Get the CFG and the Decl of this block
  C = LC->getCFG();
  D = LC->getAnalysisContext()->getDecl();

  unsigned total = 0, unreachable = 0;

  // Find CFGBlocks that were not covered by any node
  for (CFG::const_iterator I = C->begin(); I != C->end(); ++I) {
    const CFGBlock *CB = *I;
    ++total;
    // Check if the block is unreachable
    if (!reachable.count(CB)) {
      ++unreachable;
    }
  }

  // We never 'reach' the entry block, so correct the unreachable count
  unreachable--;

  // Generate the warning string
  llvm::SmallString<128> buf;
  llvm::raw_svector_ostream output(buf);
  PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
  if (Loc.isValid()) {
    output << Loc.getFilename() << " : ";

    if (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)) {
      const NamedDecl *ND = cast<NamedDecl>(D);
      output << ND;
    }
    else if (isa<BlockDecl>(D)) {
      output << "block(line:" << Loc.getLine() << ":col:" << Loc.getColumn();
    }
  }
  
  output << " -> Total CFGBlocks: " << total << " | Unreachable CFGBlocks: "
      << unreachable << " | Aborted Block: "
      << (Eng.wasBlockAborted() ? "yes" : "no")
      << " | Empty WorkList: "
      << (Eng.hasEmptyWorkList() ? "yes" : "no");

  B.EmitBasicReport("Analyzer Statistics", "Internal Statistics", output.str(),
      D->getLocation());

  // Emit warning for each block we bailed out on
  typedef CoreEngine::BlocksAborted::const_iterator AbortedIterator;
  const CoreEngine &CE = Eng.getCoreEngine();
  for (AbortedIterator I = CE.blocks_aborted_begin(),
      E = CE.blocks_aborted_end(); I != E; ++I) {
    const BlockEdge &BE =  I->first;
    const CFGBlock *Exit = BE.getDst();
    const CFGElement &CE = Exit->front();
    if (const CFGStmt *CS = dyn_cast<CFGStmt>(&CE))
      B.EmitBasicReport("Bailout Point", "Internal Statistics", "The analyzer "
          "stopped analyzing at this point", CS->getStmt()->getLocStart());
  }
}
