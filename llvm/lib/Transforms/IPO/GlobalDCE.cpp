//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
// This transform is designed to eliminate unreachable internal globals
// FIXME: GlobalDCE should update the callgraph, not destroy it!
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/CallGraph.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace {
  Statistic<> NumFunctions("globaldce","Number of functions removed");
  Statistic<> NumVariables("globaldce","Number of global variables removed");
  Statistic<> NumCPRs("globaldce", "Number of const pointer refs removed");
  Statistic<> NumConsts("globaldce", "Number of init constants removed");

  bool RemoveUnreachableFunctions(Module &M, CallGraph &CallGraph) {
    // Calculate which functions are reachable from the external functions in
    // the call graph.
    //
    std::set<CallGraphNode*> ReachableNodes(df_begin(&CallGraph),
                                            df_end(&CallGraph));

    // Loop over the functions in the module twice.  The first time is used to
    // drop references that functions have to each other before they are
    // deleted.  The second pass removes the functions that need to be removed.
    //
    std::vector<CallGraphNode*> FunctionsToDelete;   // Track unused functions
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      CallGraphNode *N = CallGraph[I];
      
      if (!ReachableNodes.count(N)) {              // Not reachable??
        I->dropAllReferences();
        N->removeAllCalledFunctions();
        FunctionsToDelete.push_back(N);
        ++NumFunctions;
      }
    }
    
    // Nothing to do if no unreachable functions have been found...
    if (FunctionsToDelete.empty()) return false;
    
    // Unreachable functions have been found and should have no references to
    // them, delete them now.
    //
    for (std::vector<CallGraphNode*>::iterator I = FunctionsToDelete.begin(),
           E = FunctionsToDelete.end(); I != E; ++I)
      delete CallGraph.removeFunctionFromModule(*I);

    // Walk the function list, removing prototypes for functions which are not
    // used.
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (I->use_size() == 0 && I->isExternal()) {
        CallGraph[I]->removeAllCalledFunctions();
        delete CallGraph.removeFunctionFromModule(I);
      }

    return true;
  }
  
  struct GlobalDCE : public Pass {
    // run - Do the GlobalDCE pass on the specified module, optionally updating
    // the specified callgraph to reflect the changes.
    //
    bool run(Module &M) {
      return RemoveUnreachableFunctions(M, getAnalysis<CallGraph>()) |
             RemoveUnreachableGlobalVariables(M);
    }

    // getAnalysisUsage - This function works on the call graph of a module.
    // It is capable of updating the call graph to reflect the new state of the
    // module.
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<CallGraph>();
    }

  private:
    std::vector<GlobalValue*> WorkList;

    inline bool RemoveIfDead(GlobalValue *GV);
    void DestroyInitializer(Constant *C);

    bool RemoveUnreachableGlobalVariables(Module &M);
    bool RemoveUnusedConstantPointerRef(GlobalValue &GV);
    bool SafeToDestroyConstant(Constant *C);
  };
  RegisterOpt<GlobalDCE> X("globaldce", "Dead Global Elimination");
}

Pass *createGlobalDCEPass() { return new GlobalDCE(); }


// RemoveIfDead - If this global value is dead, remove it from the current
// module and return true.
//
bool GlobalDCE::RemoveIfDead(GlobalValue *GV) {
  // If there is only one use of the global value, it might be a
  // ConstantPointerRef... which means that this global might actually be
  // dead.
  if (GV->use_size() == 1)
    RemoveUnusedConstantPointerRef(*GV);

  if (!GV->use_empty()) return false;

  if (GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
    // Eliminate all global variables that are unused, and that are internal, or
    // do not have an initializer.
    //
    if (GVar->hasInternalLinkage() || GVar->isExternal()) {
      Constant *Init = GVar->hasInitializer() ? GVar->getInitializer() : 0;
      GV->getParent()->getGlobalList().erase(GVar);
      ++NumVariables;

      // If there was an initializer for the global variable, try to destroy it
      // now.
      if (Init) DestroyInitializer(Init);

      // If the global variable is still on the worklist, remove it now.
      std::vector<GlobalValue*>::iterator I = std::find(WorkList.begin(),
                                                        WorkList.end(), GV);
      while (I != WorkList.end()) {
        I = WorkList.erase(I);
        I = std::find(I, WorkList.end(), GV);
      }

      return true;
    }
  } else {
    Function *F = cast<Function>(GV);
    // FIXME: TODO

  }
  return false;
}

// DestroyInitializer - A global variable was just destroyed and C is its
// initializer. If we can, destroy C and all of the constants it refers to.
//
void GlobalDCE::DestroyInitializer(Constant *C) {
  // Cannot destroy constants still being used, and cannot destroy primitive
  // types.
  if (!C->use_empty() || C->getType()->isPrimitiveType()) return;

  // If this is a CPR, the global value referred to may be dead now!  Add it to
  // the worklist.
  //
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
    WorkList.push_back(CPR->getValue());
    C->destroyConstant();
    ++NumCPRs;
  } else {
    bool DestroyContents = true;

    // As an optimization to the GlobalDCE algorithm, do attempt to destroy the
    // contents of an array of primitive types, because we know that this will
    // never succeed, and there could be a lot of them.
    //
    if (ConstantArray *CA = dyn_cast<ConstantArray>(C))
      if (CA->getType()->getElementType()->isPrimitiveType())
        DestroyContents = false;    // Nothing we can do with the subcontents

    // All other constants refer to other constants.  Destroy them if possible
    // as well.
    //
    std::vector<Value*> SubConstants;
    if (DestroyContents) SubConstants.insert(SubConstants.end(),
                                             C->op_begin(), C->op_end());

    // Destroy the actual constant...
    C->destroyConstant();
    ++NumConsts;

    if (DestroyContents) {
      // Remove duplicates from SubConstants, so that we do not call
      // DestroyInitializer on the same constant twice (the first call might
      // delete it, so this would be bad)
      //
      std::sort(SubConstants.begin(), SubConstants.end());
      SubConstants.erase(std::unique(SubConstants.begin(), SubConstants.end()),
                         SubConstants.end());

      // Loop over the subconstants, destroying them as well.
      for (unsigned i = 0, e = SubConstants.size(); i != e; ++i)
        DestroyInitializer(cast<Constant>(SubConstants[i]));
    }
  }
}

bool GlobalDCE::RemoveUnreachableGlobalVariables(Module &M) {
  bool Changed = false;
  WorkList.reserve(M.gsize());

  // Insert all of the globals into the WorkList, making sure to run
  // RemoveUnusedConstantPointerRef at least once on all globals...
  //
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Changed |= RemoveUnusedConstantPointerRef(*I);
    WorkList.push_back(I);
  }
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    Changed |= RemoveUnusedConstantPointerRef(*I);
    WorkList.push_back(I);
  }

  // Loop over the worklist, deleting global objects that we can.  Whenever we
  // delete something that might make something else dead, it gets added to the
  // worklist.
  //
  while (!WorkList.empty()) {
    GlobalValue *GV = WorkList.back();
    WorkList.pop_back();

    Changed |= RemoveIfDead(GV);
  }

  // Make sure that all memory is free'd from the worklist...
  std::vector<GlobalValue*>().swap(WorkList);
  return Changed;
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
