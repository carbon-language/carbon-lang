//===- ConstantMerge.cpp - Merge duplicate global constants ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface to a pass that merges duplicate global
// constants together into a single constant that is shared.  This is useful
// because some passes (ie TraceValues) insert a lot of string constants into
// the program, regardless of whether or not an existing string is available.
//
// Algorithm: ConstantMerge is designed to build up a map of available constants
// and eliminate duplicates when it is initialized.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumMerged("constmerge", "Number of global constants merged");

  struct ConstantMerge : public Pass {
    // run - For this pass, process all of the globals in the module,
    // eliminating duplicate constants.
    //
    bool run(Module &M);
  };

  RegisterOpt<ConstantMerge> X("constmerge","Merge Duplicate Global Constants");
}

Pass *createConstantMergePass() { return new ConstantMerge(); }


bool ConstantMerge::run(Module &M) {
  std::map<Constant*, GlobalVariable*> CMap;
  bool MadeChanges = false;
  
  for (Module::giterator GV = M.gbegin(), E = M.gend(); GV != E; ++GV)
    // Only process constants with initializers
    if (GV->isConstant() && GV->hasInitializer()) {
      Constant *Init = GV->getInitializer();

      // Check to see if the initializer is already known...
      std::map<Constant*, GlobalVariable*>::iterator I = CMap.find(Init);

      if (I == CMap.end()) {    // Nope, add it to the map
        CMap.insert(I, std::make_pair(Init, GV));
      } else if (GV->hasInternalLinkage()) {    // Yup, this is a duplicate!
        // Make all uses of the duplicate constant use the canonical version...
        GV->replaceAllUsesWith(I->second);
        
        // Delete the global value from the module... and back up iterator to
        // not skip the next global...
        GV = --M.getGlobalList().erase(GV);

        ++NumMerged;
        MadeChanges = true;
      } else if (I->second->hasInternalLinkage()) {
        // Make all uses of the duplicate constant use the canonical version...
        I->second->replaceAllUsesWith(GV);
        
        // Delete the global value from the module... and back up iterator to
        // not skip the next global...
        M.getGlobalList().erase(I->second);
        I->second = GV;

        ++NumMerged;
        MadeChanges = true;
      }
    }

  return MadeChanges;
}
