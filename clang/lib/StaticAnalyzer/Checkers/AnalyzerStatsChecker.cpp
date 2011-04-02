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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"

#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;
using namespace ento;

namespace {
class AnalyzerStatsChecker : public Checker<check::EndAnalysis> {
public:
  void checkEndAnalysis(ExplodedGraph &G, BugReporter &B,ExprEngine &Eng) const;
};
}

void AnalyzerStatsChecker::checkEndAnalysis(ExplodedGraph &G,
                                            BugReporter &B,
                                            ExprEngine &Eng) const {
  const CFG *C  = 0;
  const Decl *D = 0;
  const LocationContext *LC = 0;
  const SourceManager &SM = B.getSourceManager();
  llvm::SmallPtrSet<const CFGBlock*, 256> reachable;

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
      << unreachable << " | Exhausted Block: "
      << (Eng.wasBlocksExhausted() ? "yes" : "no")
      << " | Empty WorkList: "
      << (Eng.hasEmptyWorkList() ? "yes" : "no");

  B.EmitBasicReport("Analyzer Statistics", "Internal Statistics", output.str(),
      D->getLocation());

  // Emit warning for each block we bailed out on
  typedef CoreEngine::BlocksExhausted::const_iterator ExhaustedIterator;
  const CoreEngine &CE = Eng.getCoreEngine();
  for (ExhaustedIterator I = CE.blocks_exhausted_begin(),
      E = CE.blocks_exhausted_end(); I != E; ++I) {
    const BlockEdge &BE =  I->first;
    const CFGBlock *Exit = BE.getDst();
    const CFGElement &CE = Exit->front();
    if (const CFGStmt *CS = dyn_cast<CFGStmt>(&CE))
      B.EmitBasicReport("Bailout Point", "Internal Statistics", "The analyzer "
          "stopped analyzing at this point", CS->getStmt()->getLocStart());
  }
}

void ento::registerAnalyzerStatsChecker(CheckerManager &mgr) {
  mgr.registerChecker<AnalyzerStatsChecker>();
}
