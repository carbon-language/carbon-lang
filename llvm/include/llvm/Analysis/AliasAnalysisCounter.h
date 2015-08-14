//===- AliasAnalysisCounter.h - Alias Analysis Query Counter ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This declares an alias analysis which counts and prints queries made
/// through it. By inserting this between other AAs you can track when specific
/// layers of LLVM's AA get queried.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASANALYSISCOUNTER_H
#define LLVM_ANALYSIS_ALIASANALYSISCOUNTER_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

  class AliasAnalysisCounter : public ModulePass, public AliasAnalysis {
    unsigned No, May, Partial, Must;
    unsigned NoMR, JustRef, JustMod, MR;
    Module *M;
  public:
    static char ID; // Class identification, replacement for typeinfo
    AliasAnalysisCounter() : ModulePass(ID) {
      initializeAliasAnalysisCounterPass(*PassRegistry::getPassRegistry());
      No = May = Partial = Must = 0;
      NoMR = JustRef = JustMod = MR = 0;
    }

    void printLine(const char *Desc, unsigned Val, unsigned Sum) {
      errs() <<  "  " << Val << " " << Desc << " responses ("
             << Val*100/Sum << "%)\n";
    }
    ~AliasAnalysisCounter() override {
      unsigned AASum = No+May+Partial+Must;
      unsigned MRSum = NoMR+JustRef+JustMod+MR;
      if (AASum + MRSum) { // Print a report if any counted queries occurred...
        errs() << "\n===== Alias Analysis Counter Report =====\n"
               << "  Analysis counted:\n"
               << "  " << AASum << " Total Alias Queries Performed\n";
        if (AASum) {
          printLine("no alias",     No, AASum);
          printLine("may alias",   May, AASum);
          printLine("partial alias", Partial, AASum);
          printLine("must alias", Must, AASum);
          errs() << "  Alias Analysis Counter Summary: " << No*100/AASum << "%/"
                 << May*100/AASum << "%/"
                 << Partial*100/AASum << "%/"
                 << Must*100/AASum<<"%\n\n";
        }

        errs() << "  " << MRSum << " Total MRI_Mod/MRI_Ref Queries Performed\n";
        if (MRSum) {
          printLine("no mod/ref",    NoMR, MRSum);
          printLine("ref",        JustRef, MRSum);
          printLine("mod",        JustMod, MRSum);
          printLine("mod/ref",         MR, MRSum);
          errs() << "  MRI_Mod/MRI_Ref Analysis Counter Summary: "
                 << NoMR * 100 / MRSum << "%/" << JustRef * 100 / MRSum << "%/"
                 << JustMod * 100 / MRSum << "%/" << MR * 100 / MRSum
                 << "%\n\n";
        }
      }
    }

    bool runOnModule(Module &M) override {
      this->M = &M;
      InitializeAliasAnalysis(this, &M.getDataLayout());
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    void *getAdjustedAnalysisPointer(AnalysisID PI) override {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
    
    // FIXME: We could count these too...
    bool pointsToConstantMemory(const MemoryLocation &Loc,
                                bool OrLocal) override {
      return getAnalysis<AliasAnalysis>().pointsToConstantMemory(Loc, OrLocal);
    }

    // Forwarding functions: just delegate to a real AA implementation, counting
    // the number of responses...
    AliasResult alias(const MemoryLocation &LocA,
                      const MemoryLocation &LocB) override;

    ModRefInfo getModRefInfo(ImmutableCallSite CS,
                             const MemoryLocation &Loc) override;
    ModRefInfo getModRefInfo(ImmutableCallSite CS1,
                             ImmutableCallSite CS2) override {
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }
  };

  //===--------------------------------------------------------------------===//
  //
  // createAliasAnalysisCounterPass - This pass counts alias queries and how the
  // alias analysis implementation responds.
  //
  ModulePass *createAliasAnalysisCounterPass();

}

#endif
