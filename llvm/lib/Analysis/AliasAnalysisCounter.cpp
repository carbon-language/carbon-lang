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
  unsigned No = 0, May = 0, Must = 0;

  struct AliasAnalysisCounter : public Pass, public AliasAnalysis {
    bool run(Module &M) { return false; }
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }

    Result count(Result R) {
      switch (R) {
      default: assert(0 && "Unknown alias type!");
      case NoAlias:   No++; return NoAlias;
      case MayAlias:  May++; return MayAlias;
      case MustAlias: Must++; return MustAlias;
      }
    }
    
    // Forwarding functions: just delegate to a real AA implementation, counting
    // the number of responses...
    Result alias(const Value *V1, const Value *V2) {
      return count(getAnalysis<AliasAnalysis>().alias(V1, V2));
    }
    Result canCallModify(const CallInst &CI, const Value *Ptr) {
      return count(getAnalysis<AliasAnalysis>().canCallModify(CI, Ptr));
    }
    Result canInvokeModify(const InvokeInst &I, const Value *Ptr) {
      return count(getAnalysis<AliasAnalysis>().canInvokeModify(I, Ptr));
    }
  };

  RegisterOpt<AliasAnalysisCounter>
  X("count-aa", "Count Alias Analysis Query Responses");
  RegisterAnalysisGroup<AliasAnalysis, AliasAnalysisCounter> Y;


  struct ResultPrinter {
    ~ResultPrinter() {
      unsigned Sum = No+May+Must;
      if (Sum) {            // Print a report if any counted queries occurred...
        std::cerr
          << "\n===== Alias Analysis Counter Report =====\n"
          << "  " << Sum << " Total Alias Queries Performed\n"
          << "  " << No << " no alias responses (" << No*100/Sum << "%)\n"
          << "  " << May << " may alias responses (" << May*100/Sum << "%)\n"
          << "  " << Must << " must alias responses (" <<Must*100/Sum<<"%)\n"
          << "  Alias Analysis Counter Summary: " << No*100/Sum << "%/"
          << May*100/Sum << "%/" << Must*100/Sum<<"%\n\n";
      }
    }
  } RP;
}
