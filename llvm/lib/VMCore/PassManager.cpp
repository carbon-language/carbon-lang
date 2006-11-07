//===- PassManager.cpp - LLVM Pass Infrastructure Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Manager infrastructure. 
//
//===----------------------------------------------------------------------===//


#include "llvm/PassManager.h"
#include "llvm/Function.h"
#include "llvm/Module.h"

using namespace llvm;

/// BasicBlockPassManager implementation

/// Add pass P into PassVector and return TRUE. If this pass is not
/// manageable by this manager then return FALSE.
bool
BasicBlockPassManager_New::addPass (Pass *P) {

  BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
  if (!BP)
    return false;

  // TODO: Check if it suitable to manage P using this BasicBlockPassManager
  // or we need another instance of BasicBlockPassManager

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

///////////////////////////////////////////////////////////////////////////////
// FunctionPassManager

/// Add pass P into the pass manager queue. If P is a BasicBlockPass then
/// either use it into active basic block pass manager or create new basic
/// block pass manager to handle pass P.
bool
FunctionPassManager_New::addPass (Pass *P) {

  // If P is a BasicBlockPass then use BasicBlockPassManager_New.
  if (BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P)) {

    if (!activeBBPassManager
        || !activeBBPassManager->addPass(BP)) {

      activeBBPassManager = new BasicBlockPassManager_New();

      PassVector.push_back(activeBBPassManager);
      assert (!activeBBPassManager->addPass(BP) &&
              "Unable to add Pass");
    }
    return true;
  }

  FunctionPass *FP = dynamic_cast<FunctionPass *>(P);
  if (!FP)
    return false;

  // TODO: Check if it suitable to manage P using this FunctionPassManager
  // or we need another instance of FunctionPassManager

  PassVector.push_back(FP);
  activeBBPassManager = NULL;
  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnFunction method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
FunctionPassManager_New::runOnModule(Module &M) {

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
/// then use FunctionPassManager_New to manage it. Return FALSE if P
/// is not manageable by this manager.
bool
ModulePassManager_New::addPass (Pass *P) {

  // If P is FunctionPass then use function pass maanager.
  if (FunctionPass *FP = dynamic_cast<FunctionPass*>(P)) {

    activeFunctionPassManager = NULL;

    if (!activeFunctionPassManager
        || !activeFunctionPassManager->addPass(P)) {

      activeFunctionPassManager = new FunctionPassManager_New();

      PassVector.push_back(activeFunctionPassManager);
      assert (!activeFunctionPassManager->addPass(FP) &&
              "Unable to add Pass");
    }
    return true;
  }

  ModulePass *MP = dynamic_cast<ModulePass *>(P);
  if (!MP)
    return false;

  // TODO: Check if it suitable to manage P using this ModulePassManager
  // or we need another instance of ModulePassManager

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
PassManager_New::schedulePasses() {
  /* TODO */
}

/// Add pass P to the queue of passes to run.
void
PassManager_New::add(Pass *P) {
  /* TODO */
}

// PassManager_New implementation
/// Add P into active pass manager or use new module pass manager to
/// manage it.
bool
PassManager_New::addPass (Pass *P) {

  if (!activeManager) {
    activeManager = new ModulePassManager_New();
    PassManagers.push_back(activeManager);
  }

  return activeManager->addPass(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool
PassManager_New::run(Module &M) {

  schedulePasses();
  bool Changed = false;
  for (std::vector<ModulePassManager_New *>::iterator itr = PassManagers.begin(),
         e = PassManagers.end(); itr != e; ++itr) {
    ModulePassManager_New *pm = *itr;
    Changed |= pm->runOnModule(M);
  }
  return Changed;
}
