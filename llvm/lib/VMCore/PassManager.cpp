//===- PassManager.cpp - LLVM Pass Infrastructure Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Manager infrastructure. 
//
//===----------------------------------------------------------------------===//


#include "llvm/PassManager.h"
#include "llvm/Module.h"

using namespace llvm;

namespace llvm {

/// BasicBlockPassManager_New manages BasicBlockPass. It batches all the
/// pass together and sequence them to process one basic block before
/// processing next basic block.
class BasicBlockPassManager_New : public CommonPassManagerImpl {

public:
  BasicBlockPassManager_New() { }

  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the function, and if so, return true.
  bool runOnFunction(Function &F);

private:
  // Collection of pass that are managed by this manager
  std::vector<Pass *> PassVector;
};

/// FunctionPassManagerImpl_New manages FunctionPasses and BasicBlockPassManagers.
/// It batches all function passes and basic block pass managers together and
/// sequence them to process one function at a time before processing next
/// function.
class FunctionPassManagerImpl_New : public CommonPassManagerImpl {
public:
  FunctionPassManagerImpl_New(ModuleProvider *P) { /* TODO */ }
  FunctionPassManagerImpl_New() { 
    activeBBPassManager = NULL;
  }
  ~FunctionPassManagerImpl_New() { /* TODO */ };
 
  /// add - Add a pass to the queue of passes to run.  This passes
  /// ownership of the Pass to the PassManager.  When the
  /// PassManager_X is destroyed, the pass will be destroyed as well, so
  /// there is no need to delete the pass. (TODO delete passes.)
  /// This implies that all passes MUST be allocated with 'new'.
  void add(Pass *P) { /* TODO*/  }

  /// Add pass into the pass manager queue.
  bool addPass(Pass *P);

  /// Execute all of the passes scheduled for execution.  Keep
  /// track of whether any of the passes modifies the function, and if
  /// so, return true.
  bool runOnModule(Module &M);

private:
  // Collection of pass that are manged by this manager
  std::vector<Pass *> PassVector;
 
  // Active Pass Managers
  BasicBlockPassManager_New *activeBBPassManager;
};

/// ModulePassManager_New manages ModulePasses and function pass managers.
/// It batches all Module passes  passes and function pass managers together and
/// sequence them to process one module.
class ModulePassManager_New : public CommonPassManagerImpl {
 
public:
  ModulePassManager_New() { activeFunctionPassManager = NULL; }
  
  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);
  
private:
  // Collection of pass that are managed by this manager
  std::vector<Pass *> PassVector;
  
  // Active Pass Manager
  FunctionPassManagerImpl_New *activeFunctionPassManager;
};

/// PassManager_New manages ModulePassManagers
class PassManagerImpl_New : public CommonPassManagerImpl {

public:

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  void add(Pass *P);
 
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool run(Module &M);

private:

  /// Add a pass into a passmanager queue. This is used by schedulePasses
  bool addPass(Pass *p);

  /// Schedule all passes collected in pass queue using add(). Add all the
  /// schedule passes into various manager's queue using addPass().
  void schedulePasses();

  // Collection of pass managers
  std::vector<ModulePassManager_New *> PassManagers;

  // Collection of pass that are not yet scheduled
  std::vector<Pass *> PassVector;
  
  // Active Pass Manager
  ModulePassManager_New *activeManager;
};

} // End of llvm namespace

// CommonPassManagerImpl implementation

/// Return true IFF pass P's required analysis set does not required new
/// manager.
bool CommonPassManagerImpl::manageablePass(Pass *P) {

  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);

  // If this pass is not preserving information that is required by the other passes
  // managed by this manager then use new manager
  // TODO
  return true;
}

/// Return true IFF AnalysisID AID is currently available.
bool CommonPassManagerImpl::analysisCurrentlyAvailable(AnalysisID AID) {

  // TODO
  return false;
}

/// Augment RequiredSet by adding analysis required by pass P.
void CommonPassManagerImpl::noteDownRequiredAnalysis(Pass *P) {

  // TODO
}

/// Remove AnalysisID from the RequiredSet
void CommonPassManagerImpl::removeAnalysis(AnalysisID AID) {

  // TODO
}

/// Remove Analyss not preserved by Pass P
void CommonPassManagerImpl::removeNotPreservedAnalysis(Pass *P) {

  // TODO
}

/// BasicBlockPassManager implementation

/// Add pass P into PassVector and return true. If this pass is not
/// manageable by this manager then return false.
bool
BasicBlockPassManager_New::addPass(Pass *P) {

  BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
  if (!BP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  // Take a note of analysis required by this pass.
  noteDownRequiredAnalysis(P);

  // Add pass
  PassVector.push_back(BP);
  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnBasicBlock method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
BasicBlockPassManager_New::runOnFunction(Function &F) {

  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = PassVector.begin(),
           e = PassVector.end(); itr != e; ++itr) {
      Pass *P = *itr;
      BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
      Changed |= BP->runOnBasicBlock(*I);
    }
  return Changed;
}

// FunctionPassManager_New implementation
/// Create new Function pass manager
FunctionPassManager_New::FunctionPassManager_New() {
  FPM = new FunctionPassManagerImpl_New();
}

/// add - Add a pass to the queue of passes to run.  This passes
/// ownership of the Pass to the PassManager.  When the
/// PassManager_X is destroyed, the pass will be destroyed as well, so
/// there is no need to delete the pass. (TODO delete passes.)
/// This implies that all passes MUST be allocated with 'new'.
void 
FunctionPassManager_New::add(Pass *P) { 
  FPM->add(P);
}

/// Execute all of the passes scheduled for execution.  Keep
/// track of whether any of the passes modifies the function, and if
/// so, return true.
bool 
FunctionPassManager_New::runOnModule(Module &M) {
  return FPM->runOnModule(M);
}

// FunctionPassManagerImpl_New implementation

// FunctionPassManager

/// Add pass P into the pass manager queue. If P is a BasicBlockPass then
/// either use it into active basic block pass manager or create new basic
/// block pass manager to handle pass P.
bool
FunctionPassManagerImpl_New::addPass(Pass *P) {

  // If P is a BasicBlockPass then use BasicBlockPassManager_New.
  if (BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P)) {

    if (!activeBBPassManager
        || !activeBBPassManager->addPass(BP)) {

      activeBBPassManager = new BasicBlockPassManager_New();

      PassVector.push_back(activeBBPassManager);
      if (!activeBBPassManager->addPass(BP))
        assert(0 && "Unable to add Pass");
    }
    return true;
  }

  FunctionPass *FP = dynamic_cast<FunctionPass *>(P);
  if (!FP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  // Take a note of analysis required by this pass.
  noteDownRequiredAnalysis(P);

  PassVector.push_back(FP);
  activeBBPassManager = NULL;
  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnFunction method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
FunctionPassManagerImpl_New::runOnModule(Module &M) {

  bool Changed = false;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = PassVector.begin(),
           e = PassVector.end(); itr != e; ++itr) {
      Pass *P = *itr;
      FunctionPass *FP = dynamic_cast<FunctionPass*>(P);
      Changed |= FP->runOnFunction(*I);
    }
  return Changed;
}


// ModulePassManager implementation

/// Add P into pass vector if it is manageble. If P is a FunctionPass
/// then use FunctionPassManagerImpl_New to manage it. Return false if P
/// is not manageable by this manager.
bool
ModulePassManager_New::addPass(Pass *P) {

  // If P is FunctionPass then use function pass maanager.
  if (FunctionPass *FP = dynamic_cast<FunctionPass*>(P)) {

    activeFunctionPassManager = NULL;

    if (!activeFunctionPassManager
        || !activeFunctionPassManager->addPass(P)) {

      activeFunctionPassManager = new FunctionPassManagerImpl_New();

      PassVector.push_back(activeFunctionPassManager);
      if (!activeFunctionPassManager->addPass(FP))
        assert(0 && "Unable to add pass");
    }
    return true;
  }

  ModulePass *MP = dynamic_cast<ModulePass *>(P);
  if (!MP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  // Take a note of analysis required by this pass.
  noteDownRequiredAnalysis(P);

  PassVector.push_back(MP);
  activeFunctionPassManager = NULL;
  return true;
}


/// Execute all of the passes scheduled for execution by invoking 
/// runOnModule method.  Keep track of whether any of the passes modifies 
/// the module, and if so, return true.
bool
ModulePassManager_New::runOnModule(Module &M) {
  bool Changed = false;
  for (std::vector<Pass *>::iterator itr = PassVector.begin(),
         e = PassVector.end(); itr != e; ++itr) {
    Pass *P = *itr;
    ModulePass *MP = dynamic_cast<ModulePass*>(P);
    Changed |= MP->runOnModule(M);
  }
  return Changed;
}

/// Schedule all passes from the queue by adding them in their
/// respective manager's queue. 
void
PassManagerImpl_New::schedulePasses() {
  /* TODO */
}

/// Add pass P to the queue of passes to run.
void
PassManagerImpl_New::add(Pass *P) {
  /* TODO */
}

// PassManager_New implementation
/// Add P into active pass manager or use new module pass manager to
/// manage it.
bool
PassManagerImpl_New::addPass(Pass *P) {

  if (!activeManager) {
    activeManager = new ModulePassManager_New();
    PassManagers.push_back(activeManager);
  }

  return activeManager->addPass(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool
PassManagerImpl_New::run(Module &M) {

  schedulePasses();
  bool Changed = false;
  for (std::vector<ModulePassManager_New *>::iterator itr = PassManagers.begin(),
         e = PassManagers.end(); itr != e; ++itr) {
    ModulePassManager_New *pm = *itr;
    Changed |= pm->runOnModule(M);
  }
  return Changed;
}

/// Create new pass manager
PassManager_New::PassManager_New() {
  PM = new PassManagerImpl_New();
}

/// add - Add a pass to the queue of passes to run.  This passes ownership of
/// the Pass to the PassManager.  When the PassManager is destroyed, the pass
/// will be destroyed as well, so there is no need to delete the pass.  This
/// implies that all passes MUST be allocated with 'new'.
void 
PassManager_New::add(Pass *P) {
  PM->add(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool
PassManager_New::run(Module &M) {
  return PM->run(M);
}

