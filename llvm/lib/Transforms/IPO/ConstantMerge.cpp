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
using namespace llvm;

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

Pass *llvm::createConstantMergePass() { return new ConstantMerge(); }

bool ConstantMerge::run(Module &M) {
  std::map<Constant*, GlobalVariable*> CMap;

  // Replacements - This vector contains a list of replacements to perform.
  std::vector<std::pair<GlobalVariable*, GlobalVariable*> > Replacements;

  // First pass: identify all globals that can be merged together, filling in
  // the Replacements vector.  We cannot do the replacement in this pass because
  // doing so may cause initializers of other globals to be rewritten,
  // invalidating the Constant* pointers in CMap.
  //
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
        Replacements.push_back(std::make_pair(GV, I->second));
      } else if (I->second->hasInternalLinkage()) {
        // Make all uses of the duplicate constant use the canonical version...
        Replacements.push_back(std::make_pair(I->second, GV));
        I->second = GV;
      }
    }

  if (Replacements.empty()) return false;
  CMap.clear();

  // Now that we have figured out which replacements must be made, do them all
  // now.  This avoid invalidating the pointers in CMap, which are unneeded now.
  for (unsigned i = 0, e = Replacements.size(); i != e; ++i) {
    // Eliminate any uses of the dead global...
    Replacements[i].first->replaceAllUsesWith(Replacements[i].second);
    
    // Delete the global value from the module...
    M.getGlobalList().erase(Replacements[i].first);
  }
  
  NumMerged += Replacements.size();
  return true;
}
