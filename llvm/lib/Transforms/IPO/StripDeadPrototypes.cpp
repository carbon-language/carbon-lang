//===-- StripDeadPrototypes.cpp - Removed unused function declarations ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions in the input module, looking for 
// dead declarations and removes them.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <vector>
using namespace llvm;

STATISTIC(NumDeadPrototypes, "Number of dead prototypes removed");

namespace {

/// @brief Pass to remove unused function declarations.
class VISIBILITY_HIDDEN StripDeadPrototypesPass : public ModulePass {
public:
  StripDeadPrototypesPass() { }
  virtual bool runOnModule(Module &M);
};
RegisterPass<StripDeadPrototypesPass> X("strip-dead-prototypes", 
                                        "Strip Unused Function Prototypes");

} // end anonymous namespace

bool StripDeadPrototypesPass::runOnModule(Module &M) {
  // Collect all the functions we want to erase
  std::vector<Function*> FuncsToErase;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isDeclaration() &&         // Function must be only a prototype
        I->use_empty()) {             // Function must not be used
      FuncsToErase.push_back(&(*I));
    }

  // Erase the functions
  for (std::vector<Function*>::iterator I = FuncsToErase.begin(), 
       E = FuncsToErase.end(); I != E; ++I )
    (*I)->eraseFromParent();
  
  // Increment the statistic
  NumDeadPrototypes += FuncsToErase.size();

  // Return an indication of whether we changed anything or not.
  return !FuncsToErase.empty();
}

ModulePass *llvm::createStripDeadPrototypesPass() {
  return new StripDeadPrototypesPass();
}
