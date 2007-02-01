//===- llvm/PassManager.h - Pass Inftrastructre classes  --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLVM Pass Manager infrastructure. 
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"

using namespace llvm;
class llvm::PMDataManager;
class llvm::PMStack;

//===----------------------------------------------------------------------===//
// Overview:
// The Pass Manager Infrastructure manages passes. It's responsibilities are:
// 
//   o Manage optimization pass execution order
//   o Make required Analysis information available before pass P is run
//   o Release memory occupied by dead passes
//   o If Analysis information is dirtied by a pass then regenerate Analysis 
//     information before it is consumed by another pass.
//
// Pass Manager Infrastructure uses multiple pass managers.  They are
// PassManager, FunctionPassManager, MPPassManager, FPPassManager, BBPassManager.
// This class hierarcy uses multiple inheritance but pass managers do not derive
// from another pass manager.
//
// PassManager and FunctionPassManager are two top-level pass manager that
// represents the external interface of this entire pass manager infrastucture.
//
// Important classes :
//
// [o] class PMTopLevelManager;
//
// Two top level managers, PassManager and FunctionPassManager, derive from 
// PMTopLevelManager. PMTopLevelManager manages information used by top level 
// managers such as last user info.
//
// [o] class PMDataManager;
//
// PMDataManager manages information, e.g. list of available analysis info, 
// used by a pass manager to manage execution order of passes. It also provides
// a place to implement common pass manager APIs. All pass managers derive from
// PMDataManager.
//
// [o] class BBPassManager : public FunctionPass, public PMDataManager;
//
// BBPassManager manages BasicBlockPasses.
//
// [o] class FunctionPassManager;
//
// This is a external interface used by JIT to manage FunctionPasses. This
// interface relies on FunctionPassManagerImpl to do all the tasks.
//
// [o] class FunctionPassManagerImpl : public ModulePass, PMDataManager,
//                                     public PMTopLevelManager;
//
// FunctionPassManagerImpl is a top level manager. It manages FPPassManagers
//
// [o] class FPPassManager : public ModulePass, public PMDataManager;
//
// FPPassManager manages FunctionPasses and BBPassManagers
//
// [o] class MPPassManager : public Pass, public PMDataManager;
//
// MPPassManager manages ModulePasses and FPPassManagers
//
// [o] class PassManager;
//
// This is a external interface used by various tools to manages passes. It
// relies on PassManagerImpl to do all the tasks.
//
// [o] class PassManagerImpl : public Pass, public PMDataManager,
//                             public PMDTopLevelManager
//
// PassManagerImpl is a top level pass manager responsible for managing
// MPPassManagers.
//===----------------------------------------------------------------------===//

namespace llvm {

/// FunctionPassManager and PassManager, two top level managers, serve 
/// as the public interface of pass manager infrastructure.
enum TopLevelManagerType {
  TLM_Function,  // FunctionPassManager
  TLM_Pass       // PassManager
};
    
//===----------------------------------------------------------------------===//
// PMTopLevelManager
//
/// PMTopLevelManager manages LastUser info and collects common APIs used by
/// top level pass managers.
class PMTopLevelManager {
public:

  virtual unsigned getNumContainedManagers() {
    return PassManagers.size();
  }

  /// Schedule pass P for execution. Make sure that passes required by
  /// P are run before P is run. Update analysis info maintained by
  /// the manager. Remove dead passes. This is a recursive function.
  void schedulePass(Pass *P);

  /// This is implemented by top level pass manager and used by 
  /// schedulePass() to add analysis info passes that are not available.
  virtual void addTopLevelPass(Pass  *P) = 0;

  /// Set pass P as the last user of the given analysis passes.
  void setLastUser(std::vector<Pass *> &AnalysisPasses, Pass *P);

  /// Collect passes whose last user is P
  void collectLastUses(std::vector<Pass *> &LastUses, Pass *P);

  /// Find the pass that implements Analysis AID. Search immutable
  /// passes and all pass managers. If desired pass is not found
  /// then return NULL.
  Pass *findAnalysisPass(AnalysisID AID);

  PMTopLevelManager(enum TopLevelManagerType t);
  virtual ~PMTopLevelManager(); 

  /// Add immutable pass and initialize it.
  inline void addImmutablePass(ImmutablePass *P) {
    P->initializePass();
    ImmutablePasses.push_back(P);
  }

  inline std::vector<ImmutablePass *>& getImmutablePasses() {
    return ImmutablePasses;
  }

  void addPassManager(Pass *Manager) {
    PassManagers.push_back(Manager);
  }

  // Add Manager into the list of managers that are not directly
  // maintained by this top level pass manager
  inline void addIndirectPassManager(PMDataManager *Manager) {
    IndirectPassManagers.push_back(Manager);
  }

  // Print passes managed by this top level manager.
  void dumpPasses() const;
  void dumpArguments() const;

  void initializeAllAnalysisInfo();

  // Active Pass Managers
  PMStack activeStack;

protected:
  
  /// Collection of pass managers
  std::vector<Pass *> PassManagers;

private:

  /// Collection of pass managers that are not directly maintained
  /// by this pass manager
  std::vector<PMDataManager *> IndirectPassManagers;

  // Map to keep track of last user of the analysis pass.
  // LastUser->second is the last user of Lastuser->first.
  std::map<Pass *, Pass *> LastUser;

  /// Immutable passes are managed by top level manager.
  std::vector<ImmutablePass *> ImmutablePasses;
};


  
//===----------------------------------------------------------------------===//
// PMDataManager

/// PMDataManager provides the common place to manage the analysis data
/// used by pass managers.
class PMDataManager {
public:
  PMDataManager(int Depth) : TPM(NULL), Depth(Depth) {
    initializeAnalysisInfo();
  }

  virtual ~PMDataManager();

  /// Return true IFF pass P's required analysis set does not required new
  /// manager.
  bool manageablePass(Pass *P);

  /// Augment AvailableAnalysis by adding analysis made available by pass P.
  void recordAvailableAnalysis(Pass *P);

  /// Remove Analysis that is not preserved by the pass
  void removeNotPreservedAnalysis(Pass *P);
  
  /// Remove dead passes
  void removeDeadPasses(Pass *P, std::string &Msg);

  /// Add pass P into the PassVector. Update 
  /// AvailableAnalysis appropriately if ProcessAnalysis is true.
  void add(Pass *P, bool ProcessAnalysis = true);

  /// Initialize available analysis information.
  void initializeAnalysisInfo() { 
    TransferLastUses.clear();
    AvailableAnalysis.clear();
  }

  /// Populate RequiredPasses with the analysis pass that are required by
  /// pass P.
  void collectRequiredAnalysisPasses(std::vector<Pass *> &RequiredPasses,
                                     Pass *P);

  /// All Required analyses should be available to the pass as it runs!  Here
  /// we fill in the AnalysisImpls member of the pass so that it can
  /// successfully use the getAnalysis() method to retrieve the
  /// implementations it needs.
  void initializeAnalysisImpl(Pass *P);

  /// Find the pass that implements Analysis AID. If desired pass is not found
  /// then return NULL.
  Pass *findAnalysisPass(AnalysisID AID, bool Direction);

  // Access toplevel manager
  PMTopLevelManager *getTopLevelManager() { return TPM; }
  void setTopLevelManager(PMTopLevelManager *T) { TPM = T; }

  unsigned getDepth() const { return Depth; }

  // Print routines used by debug-pass
  void dumpLastUses(Pass *P, unsigned Offset) const;
  void dumpPassArguments() const;
  void dumpPassInfo(Pass *P,  std::string &Msg1, std::string &Msg2) const;
  void dumpAnalysisSetInfo(const char *Msg, Pass *P,
                           const std::vector<AnalysisID> &Set) const;

  std::vector<Pass *>& getTransferredLastUses() {
    return TransferLastUses;
  }

  virtual unsigned getNumContainedPasses() { 
    return PassVector.size();
  }

  virtual PassManagerType getPassManagerType() { 
    assert ( 0 && "Invalid use of getPassManagerType");
    return PMT_Unknown; 
  }
protected:

  // If a FunctionPass F is the last user of ModulePass info M
  // then the F's manager, not F, records itself as a last user of M.
  // Current pass manage is requesting parent manager to record parent
  // manager as the last user of these TrransferLastUses passes.
  std::vector<Pass *> TransferLastUses;

  // Top level manager.
  PMTopLevelManager *TPM;

  // Collection of pass that are managed by this manager
  std::vector<Pass *> PassVector;

private:
  // Set of available Analysis. This information is used while scheduling 
  // pass. If a pass requires an analysis which is not not available then 
  // equired analysis pass is scheduled to run before the pass itself is 
  // scheduled to run.
  std::map<AnalysisID, Pass*> AvailableAnalysis;

  unsigned Depth;
};

//===----------------------------------------------------------------------===//
// FPPassManager
//
/// FPPassManager manages BBPassManagers and FunctionPasses.
/// It batches all function passes and basic block pass managers together and 
/// sequence them to process one function at a time before processing next 
/// function.

class FPPassManager : public ModulePass, public PMDataManager {
 
public:
  FPPassManager(int Depth) : PMDataManager(Depth) { }
  
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnFunction(Function &F);
  bool runOnModule(Module &M);

  /// doInitialization - Run all of the initializers for the function passes.
  ///
  bool doInitialization(Module &M);
  
  /// doFinalization - Run all of the initializers for the function passes.
  ///
  bool doFinalization(Module &M);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    Info.setPreservesAll();
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset);

  virtual const char *getPassName() const {
    return "Function Pass Manager";
  }

  FunctionPass *getContainedPass(unsigned N) {
    assert ( N < PassVector.size() && "Pass number out of range!");
    FunctionPass *FP = static_cast<FunctionPass *>(PassVector[N]);
    return FP;
  }

  virtual PassManagerType getPassManagerType() { 
    return PMT_FunctionPassManager; 
  }
};

}

extern void StartPassTimer(Pass *);
extern void StopPassTimer(Pass *);
