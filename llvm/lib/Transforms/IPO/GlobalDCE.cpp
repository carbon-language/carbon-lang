//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
// This transform is designed to eliminate unreachable internal globals from the
// program.  It uses an aggressive algorithm, searching out globals that are
// known to be alive.  After it finds all of the globals which are needed, it
// deletes whatever is left over.  This allows it to delete recursive chunks of
// the program which are unreachable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "Support/Statistic.h"
#include <set>

namespace {
  Statistic<> NumFunctions("globaldce","Number of functions removed");
  Statistic<> NumVariables("globaldce","Number of global variables removed");
  Statistic<> NumCPRs("globaldce", "Number of const pointer refs removed");

  struct GlobalDCE : public Pass {
    // run - Do the GlobalDCE pass on the specified module, optionally updating
    // the specified callgraph to reflect the changes.
    //
    bool run(Module &M);

  private:
    std::set<GlobalValue*> AliveGlobals;

    /// MarkGlobalIsNeeded - the specific global value as needed, and
    /// recursively mark anything that it uses as also needed.
    void GlobalIsNeeded(GlobalValue *GV);
    void MarkUsedGlobalsAsNeeded(Constant *C);

    bool RemoveUnusedConstantPointerRef(GlobalValue &GV);
    bool SafeToDestroyConstant(Constant *C);
  };
  RegisterOpt<GlobalDCE> X("globaldce", "Dead Global Elimination");
}

Pass *createGlobalDCEPass() { return new GlobalDCE(); }

bool GlobalDCE::run(Module &M) {
  bool Changed = false;
  // Loop over the module, adding globals which are obviously necessary.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Changed |= RemoveUnusedConstantPointerRef(*I);
    // Functions with external linkage are needed if they have a body
    if (I->hasExternalLinkage() && !I->isExternal())
      GlobalIsNeeded(I);
  }

  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    Changed |= RemoveUnusedConstantPointerRef(*I);
    // Externally visible globals are needed, if they have an initializer.
    if (I->hasExternalLinkage() && !I->isExternal())
      GlobalIsNeeded(I);
  }


  // Now that all globals which are needed are in the AliveGlobals set, we loop
  // through the program, deleting those which are not alive.
  //

  // The first pass is to drop initializers of global variables which are dead.
  std::vector<GlobalVariable*> DeadGlobalVars;   // Keep track of dead globals
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    if (!AliveGlobals.count(I)) {
      DeadGlobalVars.push_back(I);         // Keep track of dead globals
      I->setInitializer(0);
    }


  // The second pass drops the bodies of functions which are dead...
  std::vector<Function*> DeadFunctions;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!AliveGlobals.count(I)) {
      DeadFunctions.push_back(I);         // Keep track of dead globals
      if (!I->isExternal())
        I->deleteBody();
    }

  if (!DeadFunctions.empty()) {
    // Now that all interreferences have been dropped, delete the actual objects
    // themselves.
    for (unsigned i = 0, e = DeadFunctions.size(); i != e; ++i) {
      RemoveUnusedConstantPointerRef(*DeadFunctions[i]);
      M.getFunctionList().erase(DeadFunctions[i]);
    }
    NumFunctions += DeadFunctions.size();
    Changed = true;
  }

  if (!DeadGlobalVars.empty()) {
    for (unsigned i = 0, e = DeadGlobalVars.size(); i != e; ++i) {
      RemoveUnusedConstantPointerRef(*DeadGlobalVars[i]);
      M.getGlobalList().erase(DeadGlobalVars[i]);
    }
    NumVariables += DeadGlobalVars.size();
    Changed = true;
  }
    
  // Make sure that all memory is released
  AliveGlobals.clear();
  return Changed;
}

/// MarkGlobalIsNeeded - the specific global value as needed, and
/// recursively mark anything that it uses as also needed.
void GlobalDCE::GlobalIsNeeded(GlobalValue *G) {
  std::set<GlobalValue*>::iterator I = AliveGlobals.lower_bound(G);

  // If the global is already in the set, no need to reprocess it.
  if (I != AliveGlobals.end() && *I == G) return;

  // Otherwise insert it now, so we do not infinitely recurse
  AliveGlobals.insert(I, G);

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(G)) {
    // If this is a global variable, we must make sure to add any global values
    // referenced by the initializer to the alive set.
    if (GV->hasInitializer())
      MarkUsedGlobalsAsNeeded(GV->getInitializer());
  } else {
    // Otherwise this must be a function object.  We have to scan the body of
    // the function looking for constants and global values which are used as
    // operands.  Any operands of these types must be processed to ensure that
    // any globals used will be marked as needed.
    Function *F = cast<Function>(G);
    // For all basic blocks...
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      // For all instructions...
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        // For all operands...
        for (User::op_iterator U = I->op_begin(), E = I->op_end(); U != E; ++U)
          if (GlobalValue *GV = dyn_cast<GlobalValue>(*U))
            GlobalIsNeeded(GV);
          else if (Constant *C = dyn_cast<Constant>(*U))
            MarkUsedGlobalsAsNeeded(C);      
  }
}

void GlobalDCE::MarkUsedGlobalsAsNeeded(Constant *C) {
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C))
    GlobalIsNeeded(CPR->getValue());
  else {
    // Loop over all of the operands of the constant, adding any globals they
    // use to the list of needed globals.
    for (User::op_iterator I = C->op_begin(), E = C->op_end(); I != E; ++I)
      MarkUsedGlobalsAsNeeded(cast<Constant>(*I));
  }
}

// RemoveUnusedConstantPointerRef - Loop over all of the uses of the specified
// GlobalValue, looking for the constant pointer ref that may be pointing to it.
// If found, check to see if the constant pointer ref is safe to destroy, and if
// so, nuke it.  This will reduce the reference count on the global value, which
// might make it deader.
//
bool GlobalDCE::RemoveUnusedConstantPointerRef(GlobalValue &GV) {
  for (Value::use_iterator I = GV.use_begin(), E = GV.use_end(); I != E; ++I)
    if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(*I))
      if (SafeToDestroyConstant(CPR)) {  // Only if unreferenced...
        CPR->destroyConstant();
        ++NumCPRs;
        return true;
      }

  return false;
}

// SafeToDestroyConstant - It is safe to destroy a constant iff it is only used
// by constants itself.  Note that constants cannot be cyclic, so this test is
// pretty easy to implement recursively.
//
bool GlobalDCE::SafeToDestroyConstant(Constant *C) {
  for (Value::use_iterator I = C->use_begin(), E = C->use_end(); I != E; ++I)
    if (Constant *User = dyn_cast<Constant>(*I)) {
      if (!SafeToDestroyConstant(User)) return false;
    } else {
      return false;
    }

  return true;
}
