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
#include <vector>
#include <map>

using namespace llvm;

namespace llvm {

/// CommonPassManagerImpl helps pass manager analysis required by
/// the managed passes. It provides methods to add/remove analysis
/// available and query if certain analysis is available or not.
class CommonPassManagerImpl : public Pass {

public:

  /// Return true IFF pass P's required analysis set does not required new
  /// manager.
  bool manageablePass(Pass *P);

  Pass *getAnalysisPass(AnalysisID AID) const {

    std::map<AnalysisID, Pass*>::const_iterator I = 
      AvailableAnalysis.find(AID);

    if (I != AvailableAnalysis.end())
      return NULL;
    else
      return I->second;
  }

  /// Augment RequiredAnalysis by adding analysis required by pass P.
  void noteDownRequiredAnalysis(Pass *P);

  /// Augment AvailableAnalysis by adding analysis made available by pass P.
  void noteDownAvailableAnalysis(Pass *P);

  /// Remove Analysis that is not preserved by the pass
  void removeNotPreservedAnalysis(Pass *P);
  
  /// Remove dead passes
  void removeDeadPasses(Pass *P);

  /// Add pass P into the PassVector. Update RequiredAnalysis and
  /// AvailableAnalysis appropriately if ProcessAnalysis is true.
  void addPassToManager (Pass *P, bool ProcessAnalysis = true);

  /// Clear analysis vectors RequiredAnalysis and AvailableAnalysis.
  /// This is used before running passes managed by the manager.
  void clearAnalysis() { 
    RequiredAnalysis.clear();
    AvailableAnalysis.clear();
    LastUser.clear();
  }

  // All Required analyses should be available to the pass as it runs!  Here
  // we fill in the AnalysisImpls member of the pass so that it can
  // successfully use the getAnalysis() method to retrieve the
  // implementations it needs.
  //
 void initializeAnalysisImpl(Pass *P);

  inline std::vector<Pass *>::iterator passVectorBegin() { 
    return PassVector.begin(); 
  }

  inline std::vector<Pass *>::iterator passVectorEnd() { 
    return PassVector.end();
  }

  inline void setLastUser(Pass *P, Pass *LU) { 
    LastUser[P] = LU; 
    // TODO : Check if pass P is available.

    // Prolong live range of analyses that are needed after an analysis pass
    // is destroyed, for querying by subsequent passes
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);
    const std::vector<AnalysisID> &IDs = AnUsage.getRequiredTransitiveSet();
    for (std::vector<AnalysisID>::const_iterator I = IDs.begin(),
           E = IDs.end(); I != E; ++I) {
      Pass *AnalysisPass = getAnalysisPass(*I); // getAnalysisPassFromManager(*I);
      assert (AnalysisPass && "Analysis pass is not available");
      setLastUser(AnalysisPass, LU);
    }

  }

private:
  // Analysis required by the passes managed by this manager. This information
  // used while selecting pass manager during addPass. If a pass does not
  // preserve any analysis required by other passes managed by current
  // pass manager then new pass manager is used.
  std::vector<AnalysisID> RequiredAnalysis;

  // Set of available Analysis. This information is used while scheduling 
  // pass. If a pass requires an analysis which is not not available then 
  // equired analysis pass is scheduled to run before the pass itself is 
  // scheduled to run.
  std::map<AnalysisID, Pass*> AvailableAnalysis;

  // Map to keep track of last user of the analysis pass.
  // LastUser->second is the last user of Lastuser->first.
  std::map<Pass *, Pass *> LastUser;

  // Collection of pass that are managed by this manager
  std::vector<Pass *> PassVector;
};

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

  /// Return true IFF AnalysisID AID is currently available.
  Pass *getAnalysisPassFromManager(AnalysisID AID);

private:
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

  /// Return true IFF AnalysisID AID is currently available.
  Pass *getAnalysisPassFromManager(AnalysisID AID);

private:
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

  /// Return true IFF AnalysisID AID is currently available.
  Pass *getAnalysisPassFromManager(AnalysisID AID);
  
private:
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

  /// Return true IFF AnalysisID AID is currently available.
  Pass *getAnalysisPassFromManager(AnalysisID AID);

private:

  /// Add a pass into a passmanager queue. This is used by schedulePasses
  bool addPass(Pass *p);

  /// Schedule pass P for execution. Make sure that passes required by
  /// P are run before P is run. Update analysis info maintained by
  /// the manager. Remove dead passes. This is a recursive function.
  void schedulePass(Pass *P);

  /// Schedule all passes collected in pass queue using add(). Add all the
  /// schedule passes into various manager's queue using addPass().
  void schedulePasses();

  // Collection of pass managers
  std::vector<ModulePassManager_New *> PassManagers;

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

  // If this pass is not preserving information that is required by the other
  // passes managed by this manager then use new manager
  if (!AnUsage.getPreservesAll()) {
    const std::vector<AnalysisID> &PreservedSet = AnUsage.getPreservedSet();
    for (std::vector<AnalysisID>::iterator I = RequiredAnalysis.begin(),
           E = RequiredAnalysis.end(); I != E; ++I) {
      if (std::find(PreservedSet.begin(), PreservedSet.end(), *I) == 
          PreservedSet.end())
        // This analysis is not preserved. Need new manager.
        return false;
    }
  }
  return true;
}

/// Augment RequiredAnalysis by adding analysis required by pass P.
void CommonPassManagerImpl::noteDownRequiredAnalysis(Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
  const std::vector<AnalysisID> &RequiredSet = AnUsage.getRequiredSet();

  // FIXME: What about duplicates ?
  RequiredAnalysis.insert(RequiredAnalysis.end(), RequiredSet.begin(), 
                          RequiredSet.end());

  initializeAnalysisImpl(P);
}

/// Augement AvailableAnalysis by adding analysis made available by pass P.
void CommonPassManagerImpl::noteDownAvailableAnalysis(Pass *P) {
                                                
  if (const PassInfo *PI = P->getPassInfo()) {
    AvailableAnalysis[PI] = P;

    //TODO This pass is the current implementation of all of the interfaces it
    //TODO implements as well.
    //TODO
    //TODO const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
    //TODO for (unsigned i = 0, e = II.size(); i != e; ++i)
    //TODO CurrentAnalyses[II[i]] = P;
  }
}

/// Remove Analyss not preserved by Pass P
void CommonPassManagerImpl::removeNotPreservedAnalysis(Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
  const std::vector<AnalysisID> &PreservedSet = AnUsage.getPreservedSet();

  for (std::map<AnalysisID, Pass*>::iterator I = AvailableAnalysis.begin(),
         E = AvailableAnalysis.end(); I != E; ++I ) {
    if (std::find(PreservedSet.begin(), PreservedSet.end(), I->first) == 
        PreservedSet.end()) {
      // Remove this analysis
      std::map<AnalysisID, Pass*>::iterator J = I++;
      AvailableAnalysis.erase(J);
    }
  }
}

/// Remove analysis passes that are not used any longer
void CommonPassManagerImpl::removeDeadPasses(Pass *P) {

  for (std::map<Pass *, Pass *>::iterator I = LastUser.begin(),
         E = LastUser.end(); I !=E; ++I) {
    if (I->second == P) {
      Pass *deadPass = I->first;
      deadPass->releaseMemory();

      std::map<AnalysisID, Pass*>::iterator Pos = 
        AvailableAnalysis.find(deadPass->getPassInfo());
      
      assert (Pos != AvailableAnalysis.end() &&
              "Pass is not available");
      AvailableAnalysis.erase(Pos);
    }
  }
}

/// Add pass P into the PassVector. Update RequiredAnalysis and
/// AvailableAnalysis appropriately if ProcessAnalysis is true.
void CommonPassManagerImpl::addPassToManager (Pass *P, 
                                              bool ProcessAnalysis) {

  if (ProcessAnalysis) {
    // Take a note of analysis required and made available by this pass
    noteDownRequiredAnalysis(P);
    noteDownAvailableAnalysis(P);

    // Remove the analysis not preserved by this pass
    removeNotPreservedAnalysis(P);
  }

  // Add pass
  PassVector.push_back(P);
}

// All Required analyses should be available to the pass as it runs!  Here
// we fill in the AnalysisImpls member of the pass so that it can
// successfully use the getAnalysis() method to retrieve the
// implementations it needs.
//
void CommonPassManagerImpl::initializeAnalysisImpl(Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
 
  for (std::vector<const PassInfo *>::const_iterator
         I = AnUsage.getRequiredSet().begin(),
         E = AnUsage.getRequiredSet().end(); I != E; ++I) {
    Pass *Impl = getAnalysisPass(*I);
    if (Impl == 0)
      assert(0 && "Analysis used but not available!");
    // TODO:  P->AnalysisImpls.push_back(std::make_pair(*I, Impl));
  }
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

  addPassToManager (BP);

  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnBasicBlock method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
BasicBlockPassManager_New::runOnFunction(Function &F) {

  bool Changed = false;
  clearAnalysis();

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = passVectorBegin(),
           e = passVectorEnd(); itr != e; ++itr) {
      Pass *P = *itr;
      
      noteDownAvailableAnalysis(P);
      BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
      Changed |= BP->runOnBasicBlock(*I);
      removeNotPreservedAnalysis(P);
      removeDeadPasses(P);
    }
  return Changed;
}

/// Return true IFF AnalysisID AID is currently available.
Pass * BasicBlockPassManager_New::getAnalysisPassFromManager(AnalysisID AID) {
  return getAnalysisPass(AID);
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
      addPassToManager(activeBBPassManager, false);
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

  addPassToManager (FP);
  activeBBPassManager = NULL;
  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnFunction method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
FunctionPassManagerImpl_New::runOnModule(Module &M) {

  bool Changed = false;
  clearAnalysis();

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = passVectorBegin(),
           e = passVectorEnd(); itr != e; ++itr) {
      Pass *P = *itr;
      
      noteDownAvailableAnalysis(P);
      FunctionPass *FP = dynamic_cast<FunctionPass*>(P);
      Changed |= FP->runOnFunction(*I);
      removeNotPreservedAnalysis(P);
      removeDeadPasses(P);
    }
  return Changed;
}

/// Return true IFF AnalysisID AID is currently available.
Pass *FunctionPassManagerImpl_New::getAnalysisPassFromManager(AnalysisID AID) {

  Pass *P = getAnalysisPass(AID);
  if (P)
    return P;

  if (activeBBPassManager && 
      activeBBPassManager->getAnalysisPass(AID) != 0)
    return activeBBPassManager->getAnalysisPass(AID);

  // TODO : Check inactive managers
  return NULL;
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
      addPassToManager(activeFunctionPassManager, false);
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

  addPassToManager(MP);
  activeFunctionPassManager = NULL;
  return true;
}


/// Execute all of the passes scheduled for execution by invoking 
/// runOnModule method.  Keep track of whether any of the passes modifies 
/// the module, and if so, return true.
bool
ModulePassManager_New::runOnModule(Module &M) {
  bool Changed = false;
  clearAnalysis();

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;

    noteDownAvailableAnalysis(P);
    ModulePass *MP = dynamic_cast<ModulePass*>(P);
    Changed |= MP->runOnModule(M);
    removeNotPreservedAnalysis(P);
    removeDeadPasses(P);
  }
  return Changed;
}

/// Return true IFF AnalysisID AID is currently available.
Pass *ModulePassManager_New::getAnalysisPassFromManager(AnalysisID AID) {

  
  Pass *P = getAnalysisPass(AID);
  if (P)
    return P;

  if (activeFunctionPassManager && 
      activeFunctionPassManager->getAnalysisPass(AID) != 0)
    return activeFunctionPassManager->getAnalysisPass(AID);

  // TODO : Check inactive managers
  return NULL;
}

/// Return true IFF AnalysisID AID is currently available.
Pass *PassManagerImpl_New::getAnalysisPassFromManager(AnalysisID AID) {

  Pass *P = NULL;
  for (std::vector<ModulePassManager_New *>::iterator itr = PassManagers.begin(),
         e = PassManagers.end(); !P && itr != e; ++itr)
    P  = (*itr)->getAnalysisPassFromManager(AID);
  return P;
}

/// Schedule pass P for execution. Make sure that passes required by
/// P are run before P is run. Update analysis info maintained by
/// the manager. Remove dead passes. This is a recursive function.
void PassManagerImpl_New::schedulePass(Pass *P) {

  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
  const std::vector<AnalysisID> &RequiredSet = AnUsage.getRequiredSet();
  for (std::vector<AnalysisID>::const_iterator I = RequiredSet.begin(),
         E = RequiredSet.end(); I != E; ++I) {

    Pass *AnalysisPass = getAnalysisPassFromManager(*I);
    if (!AnalysisPass) {
      // Schedule this analysis run first.
      AnalysisPass = (*I)->createPass();
      schedulePass(AnalysisPass);
    }
    setLastUser (AnalysisPass, P);
  }
    
  addPass(P);
}

/// Schedule all passes from the queue by adding them in their
/// respective manager's queue. 
void PassManagerImpl_New::schedulePasses() {
  for (std::vector<Pass *>::iterator I = passVectorBegin(),
         E = passVectorEnd(); I != E; ++I)
    schedulePass (*I);
}

/// Add pass P to the queue of passes to run.
void PassManagerImpl_New::add(Pass *P) {
  // Do not process Analysis now. Analysis is process while scheduling
  // the pass vector.
  addPassToManager(P, false);
}

// PassManager_New implementation
/// Add P into active pass manager or use new module pass manager to
/// manage it.
bool PassManagerImpl_New::addPass(Pass *P) {

  if (!activeManager || !activeManager->addPass(P)) {
    activeManager = new ModulePassManager_New();
    PassManagers.push_back(activeManager);
  }

  return activeManager->addPass(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool PassManagerImpl_New::run(Module &M) {

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

