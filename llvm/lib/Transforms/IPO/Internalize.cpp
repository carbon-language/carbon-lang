//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
// This pass loops over all of the functions in the input module, looking for a
// main function.  If a main function is found, all other functions are marked
// as internal.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "Support/StatisticReporter.h"

static Statistic<> NumChanged("internalize\t- Number of functions internal'd");

class InternalizePass : public Pass {
  const char *getPassName() const { return "Internalize Functions"; }

  virtual bool run(Module &M) {
    bool FoundMain = false;   // Look for a function named main...
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (I->getName() == "main" && !I->isExternal()) {
        FoundMain = true;
        break;
      }
    
    if (!FoundMain) return false;  // No main found, must be a library...

    bool Changed = false;

    // Found a main function, mark all functions not named main as internal.
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (I->getName() != "main" &&   // Leave the main function external
          !I->isExternal()) {         // Function must be defined here
        I->setInternalLinkage(true);
        Changed = true;
        ++NumChanged;
      }

    return Changed;
  }
};

Pass *createInternalizePass() {
  return new InternalizePass();
}
