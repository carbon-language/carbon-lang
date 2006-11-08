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

class ModulePassManager_New;

/// PassManagerAnalysisHelper helps pass manager analysis required by
/// the managed passes. It provides methods to add/remove analysis
/// available and query if certain analysis is available or not.
class PassManagerAnalysisHelper {

public:

  /// Return true IFF pass P's required analysis set does not required new
  /// manager.
  bool manageablePass(Pass *P);

  /// Return true IFF AnalysisID AID is currently available.
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

/// PassManager_New manages ModulePassManagers
class PassManager_New : public Pass,
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
