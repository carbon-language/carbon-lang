//===- AliasAnalysisCounter.cpp - Alias Analysis Query Counter ------------===//
//
// This file implements a pass which can be used to count how many alias queries
// are being made and how the alias analysis implementation being used responds.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"
#include <iostream>

namespace {
  class AliasAnalysisCounter : public Pass, public AliasAnalysis {
    unsigned No, May, Must;
    unsigned NoMR, JustRef, JustMod, MR;
    const char *Name;
  public:
    AliasAnalysisCounter() {
      No = May = Must = 0;
      NoMR = JustRef = JustMod = MR = 0;
    }

    void printLine(const char *Desc, unsigned Val, unsigned Sum) {
      std::cerr <<  "  " << Val << " " << Desc << " responses ("
                << Val*100/Sum << "%)\n";
    }
    ~AliasAnalysisCounter() {
      unsigned AASum = No+May+Must;
      unsigned MRSum = NoMR+JustRef+JustMod+MR;
      if (AASum + MRSum) { // Print a report if any counted queries occurred...
        std::cerr
          << "\n===== Alias Analysis Counter Report =====\n"
          << "  Analysis counted: " << Name << "\n"
          << "  " << AASum << " Total Alias Queries Performed\n";
        if (AASum) {
          printLine("no alias",     No, AASum);
          printLine("may alias",   May, AASum);
          printLine("must alias", Must, AASum);
          std::cerr
            << "  Alias Analysis Counter Summary: " << No*100/AASum << "%/"
            << May*100/AASum << "%/" << Must*100/AASum<<"%\n\n";
        }

        std::cerr
          << "  " << MRSum    << " Total Mod/Ref Queries Performed\n";
        if (MRSum) {
          printLine("no mod/ref",    NoMR, MRSum);
          printLine("ref",        JustRef, MRSum);
          printLine("mod",        JustMod, MRSum);
          printLine("mod/ref",         MR, MRSum);
          std::cerr
            << "  Mod/Ref Analysis Counter Summary: " << NoMR*100/MRSum<< "%/"
            << JustRef*100/MRSum << "%/" << JustMod*100/MRSum << "%/" 
            << MR*100/MRSum <<"%\n\n";
        }
      }
    }

    bool run(Module &M) {
      InitializeAliasAnalysis(this);
      Name = dynamic_cast<Pass*>(&getAnalysis<AliasAnalysis>())->getPassName();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }

    AliasResult count(AliasResult R) {
      switch (R) {
      default: assert(0 && "Unknown alias type!");
      case NoAlias:   No++; return NoAlias;
      case MayAlias:  May++; return MayAlias;
      case MustAlias: Must++; return MustAlias;
      }
    }
    ModRefResult count(ModRefResult R) {
      switch (R) {
      default:       assert(0 && "Unknown mod/ref type!");
      case NoModRef: NoMR++;     return NoModRef;
      case Ref:      JustRef++;  return Ref;
      case Mod:      JustMod++;  return Mod;
      case ModRef:   MR++;       return ModRef;
      }
    }
    
    // Forwarding functions: just delegate to a real AA implementation, counting
    // the number of responses...
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size) {
      return count(getAnalysis<AliasAnalysis>().alias(V1, V1Size, V2, V2Size));
    }
    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size) {
      return count(getAnalysis<AliasAnalysis>().getModRefInfo(CS, P, Size));
    }
  };

  RegisterOpt<AliasAnalysisCounter>
  X("count-aa", "Count Alias Analysis Query Responses");
  RegisterAnalysisGroup<AliasAnalysis, AliasAnalysisCounter> Y;
}
