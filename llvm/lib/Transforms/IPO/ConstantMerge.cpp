//===- ConstantMerge.cpp - Merge duplicate global constants -----------------=//
//
// This file defines the interface to a pass that merges duplicate global
// constants together into a single constant that is shared.  This is useful
// because some passes (ie TraceValues) insert a lot of string constants into
// the program, regardless of whether or not they duplicate an existing string.
//
// Algorithm: ConstantMerge is designed to build up a map of available constants
// and elminate duplicates when it is initialized.
//
// The DynamicConstantMerge method is a superset of the ConstantMerge algorithm
// that checks for each function to see if constants have been added to the
// constant pool since it was last run... if so, it processes them.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "Support/StatisticReporter.h"

namespace {
  struct ConstantMerge : public Pass {
    // run - For this pass, process all of the globals in the module,
    // eliminating duplicate constants.
    //
    bool run(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  };

Statistic<> NumMerged("constmerge\t\t- Number of global constants merged");
RegisterPass<ConstantMerge> X("constmerge", "Merge Duplicate Global Constants");
}

Pass *createConstantMergePass() { return new ConstantMerge(); }


// ConstantMerge::run - Workhorse for the pass.  This eliminates duplicate
// constants, starting at global ConstantNo, and adds vars to the map if they
// are new and unique.
//
bool ConstantMerge::run(Module &M) {
  std::map<Constant*, GlobalVariable*> CMap;
  bool MadeChanges = false;
  
  for (Module::giterator GV = M.gbegin(), E = M.gend(); GV != E; ++GV)
    if (GV->isConstant()) {  // Only process constants
      assert(GV->hasInitializer() && "Globals constants must have inits!");
      Constant *Init = GV->getInitializer();

      // Check to see if the initializer is already known...
      std::map<Constant*, GlobalVariable*>::iterator I = CMap.find(Init);

      if (I == CMap.end()) {    // Nope, add it to the map
        CMap.insert(I, std::make_pair(Init, GV));
      } else {                  // Yup, this is a duplicate!
        // Make all uses of the duplicate constant use the cannonical version...
        GV->replaceAllUsesWith(I->second);

        // Delete the global value from the module... and back up iterator to
        // not skip the next global...
        GV = --M.getGlobalList().erase(GV);

        ++NumMerged;
        MadeChanges = true;
      }
    }

  return MadeChanges;
}
