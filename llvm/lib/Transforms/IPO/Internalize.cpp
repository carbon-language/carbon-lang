//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
// This pass loops over all of the functions in the input module, looking for a
// main function.  If a main function is found, all other functions and all
// global variables with initializers are marked as internal.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumFunctions("internalize", "Number of functions internalized");
  Statistic<> NumGlobals  ("internalize", "Number of global vars internalized");

  class InternalizePass : public Pass {
    virtual bool run(Module &M) {
      Function *MainFunc = M.getMainFunction();

      if (MainFunc == 0 || MainFunc->isExternal())
        return false;  // No main found, must be a library...
      
      bool Changed = false;
      
      // Found a main function, mark all functions not named main as internal.
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (&*I != MainFunc &&          // Leave the main function external
            !I->isExternal() &&         // Function must be defined here
            !I->hasInternalLinkage()) { // Can't already have internal linkage
          I->setLinkage(GlobalValue::InternalLinkage);
          Changed = true;
          ++NumFunctions;
          DEBUG(std::cerr << "Internalizing func " << I->getName() << "\n");
        }

      // Mark all global variables with initializers as internal as well...
      for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
        if (!I->isExternal() && I->hasExternalLinkage()) {
          I->setLinkage(GlobalValue::InternalLinkage);
          Changed = true;
          ++NumGlobals;
          DEBUG(std::cerr << "Internalizing gvar " << I->getName() << "\n");
        }
      
      return Changed;
    }
  };

  RegisterOpt<InternalizePass> X("internalize", "Internalize Functions");
} // end anonymous namespace

Pass *createInternalizePass() {
  return new InternalizePass();
}
