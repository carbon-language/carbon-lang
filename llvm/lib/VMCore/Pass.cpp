//===- Pass.cpp - LLVM Pass Infrastructure Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"
#ifdef USE_OLD_PASSMANAGER
#include "PassManagerT.h"         // PassManagerT implementation
#endif
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TypeInfo.h"
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
//   AnalysisResolver Class Implementation
//

AnalysisResolver::~AnalysisResolver() {
}
void AnalysisResolver::setAnalysisResolver(Pass *P, AnalysisResolver *AR) {
  assert(P->Resolver == 0 && "Pass already in a PassManager!");
  P->Resolver = AR;
}

#ifdef USE_OLD_PASSMANAGER
//===----------------------------------------------------------------------===//
// PassManager implementation - The PassManager class is a simple Pimpl class
// that wraps the PassManagerT template.
//
PassManager::PassManager() : PM(new ModulePassManager()) {}
PassManager::~PassManager() { delete PM; }
void PassManager::add(Pass *P) {
  ModulePass *MP = dynamic_cast<ModulePass*>(P);
  assert(MP && "Not a modulepass?");
  PM->add(MP);
}
bool PassManager::run(Module &M) { return PM->runOnModule(M); }

//===----------------------------------------------------------------------===//
// FunctionPassManager implementation - The FunctionPassManager class
// is a simple Pimpl class that wraps the PassManagerT template. It
// is like PassManager, but only deals in FunctionPasses.
//
FunctionPassManager::FunctionPassManager(ModuleProvider *P) :
  PM(new FunctionPassManagerT()), MP(P) {}
FunctionPassManager::~FunctionPassManager() { delete PM; }
void FunctionPassManager::add(FunctionPass *P) { PM->add(P); }
void FunctionPassManager::add(ImmutablePass *IP) { PM->add(IP); }

/// doInitialization - Run all of the initializers for the function passes.
///
bool FunctionPassManager::doInitialization() {
  return PM->doInitialization(*MP->getModule());
}

bool FunctionPassManager::run(Function &F) {
  std::string errstr;
  if (MP->materializeFunction(&F, &errstr)) {
    cerr << "Error reading bytecode file: " << errstr << "\n";
    abort();
  }
  return PM->runOnFunction(F);
}

/// doFinalization - Run all of the initializers for the function passes.
///
bool FunctionPassManager::doFinalization() {
  return PM->doFinalization(*MP->getModule());
}


//===----------------------------------------------------------------------===//
// TimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens with
// -time-passes is enabled on the command line.
//
bool llvm::TimePassesIsEnabled = false;
static cl::opt<bool,true>
EnableTiming("time-passes", cl::location(TimePassesIsEnabled),
            cl::desc("Time each pass, printing elapsed time for each on exit"));

// createTheTimeInfo - This method either initializes the TheTimeInfo pointer to
// a non null value (if the -time-passes option is enabled) or it leaves it
// null.  It may be called multiple times.
void TimingInfo::createTheTimeInfo() {
  if (!TimePassesIsEnabled || TheTimeInfo) return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed before static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<TimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

void PMDebug::PrintArgumentInformation(const Pass *P) {
  // Print out passes in pass manager...
  if (const AnalysisResolver *PM = dynamic_cast<const AnalysisResolver*>(P)) {
    for (unsigned i = 0, e = PM->getNumContainedPasses(); i != e; ++i)
      PrintArgumentInformation(PM->getContainedPass(i));

  } else {  // Normal pass.  Print argument information...
    // Print out arguments for registered passes that are _optimizations_
    if (const PassInfo *PI = P->getPassInfo())
      if (!PI->isAnalysisGroup())
        cerr << " -" << PI->getPassArgument();
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Module *M) {
  if (PassDebugging >= Executions) {
    cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
         << P->getPassName();
    if (M) cerr << "' on Module '" << M->getModuleIdentifier() << "'\n";
    cerr << "'...\n";
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Function *F) {
  if (PassDebugging >= Executions) {
    cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
         << P->getPassName();
    if (F) cerr << "' on Function '" << F->getName();
    cerr << "'...\n";
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, BasicBlock *BB) {
  if (PassDebugging >= Executions) {
    cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
         << P->getPassName();
    if (BB) cerr << "' on BasicBlock '" << BB->getName();
    cerr << "'...\n";
  }
}

void PMDebug::PrintAnalysisSetInfo(unsigned Depth, const char *Msg,
                                   Pass *P, const std::vector<AnalysisID> &Set){
  if (PassDebugging >= Details && !Set.empty()) {
    cerr << (void*)P << std::string(Depth*2+3, ' ') << Msg << " Analyses:";
    for (unsigned i = 0; i != Set.size(); ++i) {
      if (i) cerr << ",";
      cerr << " " << Set[i]->getPassName();
    }
    cerr << "\n";
  }
}
#endif

//===----------------------------------------------------------------------===//
// Pass Implementation
//

#ifdef USE_OLD_PASSMANAGER
void ModulePass::addToPassManager(ModulePassManager *PM, AnalysisUsage &AU) {
  PM->addPass(this, AU);
}
#endif

bool Pass::mustPreserveAnalysisID(const PassInfo *AnalysisID) const {
#ifdef USE_OLD_PASSMANAGER
  return Resolver->getAnalysisToUpdate(AnalysisID) != 0;
#else
  return Resolver_New->getAnalysisToUpdate(AnalysisID, true) != 0;
#endif
}

// dumpPassStructure - Implement the -debug-passes=Structure option
void Pass::dumpPassStructure(unsigned Offset) {
  cerr << std::string(Offset*2, ' ') << getPassName() << "\n";
}

// getPassName - Use C++ RTTI to get a SOMEWHAT intelligible name for the pass.
//
const char *Pass::getPassName() const {
  if (const PassInfo *PI = getPassInfo())
    return PI->getPassName();
  return typeid(*this).name();
}

// print - Print out the internal state of the pass.  This is called by Analyze
// to print out the contents of an analysis.  Otherwise it is not necessary to
// implement this method.
//
void Pass::print(std::ostream &O,const Module*) const {
  O << "Pass::print not implemented for pass: '" << getPassName() << "'!\n";
}

// dump - call print(cerr);
void Pass::dump() const {
  print(*cerr.stream(), 0);
}

//===----------------------------------------------------------------------===//
// ImmutablePass Implementation
//
#ifdef USE_OLD_PASSMANAGER
void ImmutablePass::addToPassManager(ModulePassManager *PM, 
                                     AnalysisUsage &AU) {
  PM->addPass(this, AU);
}
#endif

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

// run - On a module, we run this pass by initializing, runOnFunction'ing once
// for every function in the module, then by finalizing.
//
bool FunctionPass::runOnModule(Module &M) {
  bool Changed = doInitialization(M);

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())      // Passes are not run on external functions!
    Changed |= runOnFunction(*I);

  return Changed | doFinalization(M);
}

// run - On a function, we simply initialize, run the function, then finalize.
//
bool FunctionPass::run(Function &F) {
  if (F.isExternal()) return false;// Passes are not run on external functions!

  bool Changed = doInitialization(*F.getParent());
  Changed |= runOnFunction(F);
  return Changed | doFinalization(*F.getParent());
}

#ifdef USE_OLD_PASSMANAGER
void FunctionPass::addToPassManager(ModulePassManager *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void FunctionPass::addToPassManager(FunctionPassManagerT *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}
#endif

//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

// To run this pass on a function, we simply call runOnBasicBlock once for each
// function.
//
bool BasicBlockPass::runOnFunction(Function &F) {
  bool Changed = doInitialization(F);
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    Changed |= runOnBasicBlock(*I);
  return Changed | doFinalization(F);
}

// To run directly on the basic block, we initialize, runOnBasicBlock, then
// finalize.
//
bool BasicBlockPass::runPass(BasicBlock &BB) {
  Function &F = *BB.getParent();
  Module &M = *F.getParent();
  bool Changed = doInitialization(M);
  Changed |= doInitialization(F);
  Changed |= runOnBasicBlock(BB);
  Changed |= doFinalization(F);
  Changed |= doFinalization(M);
  return Changed;
}

#ifdef USE_OLD_PASSMANAGER
void BasicBlockPass::addToPassManager(FunctionPassManagerT *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void BasicBlockPass::addToPassManager(BasicBlockPassManager *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}
#endif

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//
namespace {
class PassRegistrar {
  /// PassInfoMap - Keep track of the passinfo object for each registered llvm
  /// pass.
  std::map<TypeInfo, PassInfo*> PassInfoMap;
  
  /// AnalysisGroupInfo - Keep track of information for each analysis group.
  struct AnalysisGroupInfo {
    const PassInfo *DefaultImpl;
    std::set<const PassInfo *> Implementations;
    AnalysisGroupInfo() : DefaultImpl(0) {}
  };
  
  /// AnalysisGroupInfoMap - Information for each analysis group.
  std::map<const PassInfo *, AnalysisGroupInfo> AnalysisGroupInfoMap;

public:
  
  const PassInfo *GetPassInfo(const std::type_info &TI) const {
    std::map<TypeInfo, PassInfo*>::const_iterator I = PassInfoMap.find(TI);
    return I != PassInfoMap.end() ? I->second : 0;
  }
  
  void RegisterPass(PassInfo &PI) {
    bool Inserted =
      PassInfoMap.insert(std::make_pair(TypeInfo(PI.getTypeInfo()),&PI)).second;
    assert(Inserted && "Pass registered multiple times!");
  }
  
  void UnregisterPass(PassInfo &PI) {
    std::map<TypeInfo, PassInfo*>::iterator I =
      PassInfoMap.find(PI.getTypeInfo());
    assert(I != PassInfoMap.end() && "Pass registered but not in map!");
    
    // Remove pass from the map.
    PassInfoMap.erase(I);
  }
  
  void EnumerateWith(PassRegistrationListener *L) {
    for (std::map<TypeInfo, PassInfo*>::const_iterator I = PassInfoMap.begin(),
         E = PassInfoMap.end(); I != E; ++I)
      L->passEnumerate(I->second);
  }
  
  
  /// Analysis Group Mechanisms.
  void RegisterAnalysisGroup(PassInfo *InterfaceInfo,
                             const PassInfo *ImplementationInfo,
                             bool isDefault) {
    AnalysisGroupInfo &AGI = AnalysisGroupInfoMap[InterfaceInfo];
    assert(AGI.Implementations.count(ImplementationInfo) == 0 &&
           "Cannot add a pass to the same analysis group more than once!");
    AGI.Implementations.insert(ImplementationInfo);
    if (isDefault) {
      assert(AGI.DefaultImpl == 0 && InterfaceInfo->getNormalCtor() == 0 &&
             "Default implementation for analysis group already specified!");
      assert(ImplementationInfo->getNormalCtor() &&
           "Cannot specify pass as default if it does not have a default ctor");
      AGI.DefaultImpl = ImplementationInfo;
      InterfaceInfo->setNormalCtor(ImplementationInfo->getNormalCtor());
    }
  }
};
}

static ManagedStatic<PassRegistrar> PassRegistrarObj;
static std::vector<PassRegistrationListener*> *Listeners = 0;

// getPassInfo - Return the PassInfo data structure that corresponds to this
// pass...
const PassInfo *Pass::getPassInfo() const {
  if (PassInfoCache) return PassInfoCache;
  return lookupPassInfo(typeid(*this));
}

const PassInfo *Pass::lookupPassInfo(const std::type_info &TI) {
  return PassRegistrarObj->GetPassInfo(TI);
}

void RegisterPassBase::registerPass() {
  PassRegistrarObj->RegisterPass(PIObj);

  // Notify any listeners.
  if (Listeners)
    for (std::vector<PassRegistrationListener*>::iterator
           I = Listeners->begin(), E = Listeners->end(); I != E; ++I)
      (*I)->passRegistered(&PIObj);
}

void RegisterPassBase::unregisterPass() {
  PassRegistrarObj->UnregisterPass(PIObj);
}

//===----------------------------------------------------------------------===//
//                  Analysis Group Implementation Code
//===----------------------------------------------------------------------===//

// RegisterAGBase implementation
//
RegisterAGBase::RegisterAGBase(const std::type_info &Interface,
                               const std::type_info *Pass, bool isDefault)
  : RegisterPassBase(Interface),
    ImplementationInfo(0), isDefaultImplementation(isDefault) {

  InterfaceInfo = const_cast<PassInfo*>(Pass::lookupPassInfo(Interface));
  if (InterfaceInfo == 0) {
    // First reference to Interface, register it now.
    registerPass();
    InterfaceInfo = &PIObj;
  }
  assert(PIObj.isAnalysisGroup() &&
         "Trying to join an analysis group that is a normal pass!");

  if (Pass) {
    ImplementationInfo = Pass::lookupPassInfo(*Pass);
    assert(ImplementationInfo &&
           "Must register pass before adding to AnalysisGroup!");

    // Make sure we keep track of the fact that the implementation implements
    // the interface.
    PassInfo *IIPI = const_cast<PassInfo*>(ImplementationInfo);
    IIPI->addInterfaceImplemented(InterfaceInfo);
    
    PassRegistrarObj->RegisterAnalysisGroup(InterfaceInfo, IIPI, isDefault);
  }
}

void RegisterAGBase::setGroupName(const char *Name) {
  assert(InterfaceInfo->getPassName()[0] == 0 && "Interface Name already set!");
  InterfaceInfo->setPassName(Name);
}


//===----------------------------------------------------------------------===//
// PassRegistrationListener implementation
//

// PassRegistrationListener ctor - Add the current object to the list of
// PassRegistrationListeners...
PassRegistrationListener::PassRegistrationListener() {
  if (!Listeners) Listeners = new std::vector<PassRegistrationListener*>();
  Listeners->push_back(this);
}

// dtor - Remove object from list of listeners...
PassRegistrationListener::~PassRegistrationListener() {
  std::vector<PassRegistrationListener*>::iterator I =
    std::find(Listeners->begin(), Listeners->end(), this);
  assert(Listeners && I != Listeners->end() &&
         "PassRegistrationListener not registered!");
  Listeners->erase(I);

  if (Listeners->empty()) {
    delete Listeners;
    Listeners = 0;
  }
}

// enumeratePasses - Iterate over the registered passes, calling the
// passEnumerate callback on each PassInfo object.
//
void PassRegistrationListener::enumeratePasses() {
  PassRegistrarObj->EnumerateWith(this);
}

//===----------------------------------------------------------------------===//
//   AnalysisUsage Class Implementation
//

namespace {
  struct GetCFGOnlyPasses : public PassRegistrationListener {
    std::vector<AnalysisID> &CFGOnlyList;
    GetCFGOnlyPasses(std::vector<AnalysisID> &L) : CFGOnlyList(L) {}
    
    void passEnumerate(const PassInfo *P) {
      if (P->isCFGOnlyPass())
        CFGOnlyList.push_back(P);
    }
  };
}

// setPreservesCFG - This function should be called to by the pass, iff they do
// not:
//
//  1. Add or remove basic blocks from the function
//  2. Modify terminator instructions in any way.
//
// This function annotates the AnalysisUsage info object to say that analyses
// that only depend on the CFG are preserved by this pass.
//
void AnalysisUsage::setPreservesCFG() {
  // Since this transformation doesn't modify the CFG, it preserves all analyses
  // that only depend on the CFG (like dominators, loop info, etc...)
  GetCFGOnlyPasses(Preserved).enumeratePasses();
}


