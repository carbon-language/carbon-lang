//===- llvm/PassManager.h - Container for Passes ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManager class.  This class is used to hold,
// maintain, and optimize execution of Passes.  The PassManager class ensures
// that analysis results are available before a pass runs, and that Pass's are
// destroyed when the PassManager is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_H
#define LLVM_PASSMANAGER_H

#include "llvm/Pass.h"
#include <vector>

namespace llvm {

class Pass;
class ModulePass;
class Module;
class ModuleProvider;
class ModulePassManager;
class FunctionPassManagerT;
class BasicBlockPassManager;

class PassManager {
  ModulePassManager *PM;    // This is a straightforward Pimpl class
public:
  PassManager();
  ~PassManager();

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  ///
  void add(Pass *P);

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  ///
  bool run(Module &M);
};

class FunctionPass;
class ImmutablePass;
class Function;

class FunctionPassManager {
  FunctionPassManagerT *PM;    // This is a straightforward Pimpl class
  ModuleProvider *MP;
public:
  FunctionPassManager(ModuleProvider *P);
  ~FunctionPassManager();

  /// add - Add a pass to the queue of passes to run.  This passes
  /// ownership of the FunctionPass to the PassManager.  When the
  /// PassManager is destroyed, the pass will be destroyed as well, so
  /// there is no need to delete the pass.  This implies that all
  /// passes MUST be allocated with 'new'.
  ///
  void add(FunctionPass *P);

  /// add - ImmutablePasses are not FunctionPasses, so we have a
  /// special hack to get them into a FunctionPassManager.
  ///
  void add(ImmutablePass *IP);

  /// doInitialization - Run all of the initializers for the function passes.
  ///
  bool doInitialization();
  
  /// run - Execute all of the passes scheduled for execution.  Keep
  /// track of whether any of the passes modifies the function, and if
  /// so, return true.
  ///
  bool run(Function &F);
  
  /// doFinalization - Run all of the initializers for the function passes.
  ///
  bool doFinalization();
};

/// PassManagerAnalysisHelper helpes pass manager analysis required by
/// the managed passes.
class PassManagerAnalysisHelper {

public:

  /// Return TRUE IFF pass P's required analysis set does not required new
  /// manager.
  bool manageablePass(Pass *P);

  /// Return TRUE iff AnalysisID AID is currently available.
  bool analysisCurrentlyAvailable(AnalysisID AID);

  /// Augment RequiredSet by adding analysis required by pass P.
  void noteDownRequiredAnalysis(Pass *P);

  /// Remove AnalysisID from the RequiredSet
  void removeAnalysis(AnalysisID AID);

  /// Remove Analysis that is not preserved by the pass
  void removeNotPreservedAnalysis(Pass *P);
  
  /// Remove dead passes
  void removeDeadPasses() { /* TODO : Implement */ }

private:
   // Required set of analysis for the passes managed by this manager
  std::vector<AnalysisID> RequiredSet;
};

/// BasicBlockpassManager_New manages BasicBlockPass. It batches all the
/// pass together and sequence them to process one basic block before
/// processing next basic block.
class BasicBlockPassManager_New: public Pass,
                                 public PassManagerAnalysisHelper {

public:
  BasicBlockPassManager_New() { }

  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the function, and if so, return true.
  bool runOnFunction(Function &F);

private:
  // Collection of pass that are not yet scheduled
  std::vector<Pass *> PassVector;
};

/// FunctionPassManager_New manages FunctionPasses and BasicBlockPassManagers.
/// It batches all function passes and basic block pass managers together and
/// sequence them to process one function at a time before processing next
/// function.
class FunctionPassManager_New: public Pass,
                               public PassManagerAnalysisHelper {
public:
  FunctionPassManager_New(ModuleProvider *P) { /* TODO */ }
  FunctionPassManager_New() { 
    activeBBPassManager = NULL;
  }
  ~FunctionPassManager_New() { /* TODO */ };
 
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
  // Collection of pass that are not yet scheduled
  std::vector<Pass *> PassVector;
 
  // Active Pass Managers
  BasicBlockPassManager_New *activeBBPassManager;
};

/// ModulePassManager_New manages ModulePasses and function pass managers.
/// It batches all Module passes  passes and function pass managers together and
/// sequence them to process one module.
class ModulePassManager_New: public Pass,
                             public PassManagerAnalysisHelper {
 
public:
  ModulePassManager_New() { activeFunctionPassManager = NULL; }
  
  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);
  
private:
  // Collection of pass that are not yet scheduled
  std::vector<Pass *> PassVector;
  
  // Active Pass Manager
  FunctionPassManager_New *activeFunctionPassManager;
};

/// PassManager_New manages ModulePassManagers
class PassManager_New: public Pass,
                       public PassManagerAnalysisHelper {

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

} // End llvm namespace

#endif
