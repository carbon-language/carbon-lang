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

#include "llvm/Transforms/ConstantMerge.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

// mergeDuplicateConstants - Workhorse for the pass.  This eliminates duplicate
// constants, starting at global ConstantNo, and adds vars to the map if they
// are new and unique.
//
static inline 
bool mergeDuplicateConstants(Module *M, unsigned &ConstantNo,
                             std::map<Constant*, GlobalVariable*> &CMap) {
  Module::GlobalListType &GList = M->getGlobalList();
  if (GList.size() <= ConstantNo) return false;   // No new constants
  bool MadeChanges = false;
  
  for (; ConstantNo < GList.size(); ++ConstantNo) {
    GlobalVariable *GV = GList[ConstantNo];
    if (GV->isConstant()) {  // Only process constants
      assert(GV->hasInitializer() && "Globals constants must have inits!");
      Constant *Init = GV->getInitializer();

      // Check to see if the initializer is already known...
      std::map<Constant*, GlobalVariable*>::iterator I = CMap.find(Init);

      if (I == CMap.end()) {    // Nope, add it to the map
        CMap.insert(std::make_pair(Init, GV));
      } else {                  // Yup, this is a duplicate!
        // Make all uses of the duplicate constant use the cannonical version...
        GV->replaceAllUsesWith(I->second);

        // Remove and delete the global value from the module...
        delete GList.remove(GList.begin()+ConstantNo);

        --ConstantNo;  // Don't skip the next constant.
        MadeChanges = true;
      }
    }
  }
  return MadeChanges;
}

namespace {
  // FIXME: ConstantMerge should not be a FunctionPass!!!
  class ConstantMerge : public FunctionPass {
  protected:
    std::map<Constant*, GlobalVariable*> Constants;
    unsigned LastConstantSeen;
  public:
    inline ConstantMerge() : LastConstantSeen(0) {}
    
    // doInitialization - For this pass, process all of the globals in the
    // module, eliminating duplicate constants.
    //
    bool doInitialization(Module *M) {
      return ::mergeDuplicateConstants(M, LastConstantSeen, Constants);
    }
    
    bool runOnFunction(Function *) { return false; }
    
    // doFinalization - Clean up internal state for this module
    //
    bool doFinalization(Module *M) {
      LastConstantSeen = 0;
      Constants.clear();
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
  
  struct DynamicConstantMerge : public ConstantMerge {
    // runOnFunction - Check to see if any globals have been added to the 
    // global list for the module.  If so, eliminate them.
    //
    bool runOnFunction(Function *F) {
      return ::mergeDuplicateConstants(F->getParent(), LastConstantSeen,
                                       Constants);
    }
  };
}

Pass *createConstantMergePass() { return new ConstantMerge(); }
Pass *createDynamicConstantMergePass() { return new DynamicConstantMerge(); }
