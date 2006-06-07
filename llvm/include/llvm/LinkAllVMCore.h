//===- LinkAllVMCore.h - Reference All VMCore Code --------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all analysis passes for tools like analyze and
// bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKALLVMCORE_H
#define LLVM_LINKALLVMCORE_H

#include <llvm/Support/IncludeFile.h>
#include <llvm/Module.h>
#include <llvm/IntrinsicInst.h>
#include <llvm/IntrinsicInst.h>
#include <llvm/Instructions.h>
#include <llvm/Analysis/Dominators.h>
#include <llvm/Analysis/Verifier.h>

namespace {
  struct ForceVMCoreLinking {
    ForceVMCoreLinking() {
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
      (void)new llvm::FindUsedTypes();
      (void)new llvm::ScalarEvolution();
      (void)new llvm::CallTargetFinder();
      ((llvm::Function*)0)->viewCFGOnly();
      llvm::AliasSetTracker X(*(llvm::AliasAnalysis*)0);
      X.add((llvm::Value*)0, 0);  // for -print-alias-sets
    }
  } ForceVMCoreLinking;
}

#endif
