//===- LinkAllAnalyses.h - Reference All Analysis Passes --------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all analysis passes for tools like analyze and
// bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LINKALLANALYSES_H
#define LLVM_ANALYSIS_LINKALLANALYSES_H

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Function.h"
#include <cstdlib>

namespace {
  struct ForceAnalysisPassLinking {
    ForceAnalysisPassLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void)new llvm::LocalDataStructures();
      (void)new llvm::BUDataStructures();
      (void)new llvm::TDDataStructures();
      (void)new llvm::CompleteBUDataStructures();
      (void)new llvm::EquivClassGraphs();
      (void)llvm::createDataStructureStatsPass();
      (void)llvm::createDataStructureGraphCheckerPass();
      (void)llvm::createProfileLoaderPass();
      (void)llvm::createNoProfileInfoPass();
      (void)llvm::createInstCountPass();
      (void)new llvm::IntervalPartition();
      (void)new llvm::ImmediateDominators();
      (void)new llvm::PostDominatorSet();
      (void)new llvm::CallGraph();
      (void)new llvm::FindUsedTypes();
      (void)new llvm::FindUnsafePointerTypes();
      (void)new llvm::ScalarEvolution();
      ((llvm::Function*)0)->viewCFGOnly();
      llvm::AliasSetTracker X(*(llvm::AliasAnalysis*)0);
      X.add((llvm::Value*)0, 0);  // for -print-alias-sets
    }
  } ForceAnalysisPassLinking;
};

#endif
