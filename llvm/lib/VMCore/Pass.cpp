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
#include "PassManagerT.h"         // PassManagerT implementation
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TypeInfo.h"
#include <iostream>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
//   AnalysisID Class Implementation
//

// getCFGOnlyAnalyses - A wrapper around the CFGOnlyAnalyses which make it
// initializer order independent.
static std::vector<const PassInfo*> &getCFGOnlyAnalyses() {
  static std::vector<const PassInfo*> CFGOnlyAnalyses;
  return CFGOnlyAnalyses;
}

void RegisterPassBase::setOnlyUsesCFG() {
  getCFGOnlyAnalyses().push_back(&PIObj);
}

//===----------------------------------------------------------------------===//
//   AnalysisResolver Class Implementation
//

AnalysisResolver::~AnalysisResolver() {
}
void AnalysisResolver::setAnalysisResolver(Pass *P, AnalysisResolver *AR) {
  assert(P->Resolver == 0 && "Pass already in a PassManager!");
  P->Resolver = AR;
}

//===----------------------------------------------------------------------===//
//   AnalysisUsage Class Implementation
//

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
  //
  Preserved.insert(Preserved.end(),
                   getCFGOnlyAnalyses().begin(), getCFGOnlyAnalyses().end());
}


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
bool FunctionPassManager::run(Function &F) {
  std::string errstr;
  if (MP->materializeFunction(&F, &errstr)) {
    std::cerr << "Error reading bytecode file: " << errstr << "\n";
    abort();
  }
  return PM->run(F);
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
  static TimingInfo TTI;
  TheTimeInfo = &TTI;
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
        std::cerr << " -" << PI->getPassArgument();
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Module *M) {
  if (PassDebugging >= Executions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
              << P->getPassName();
    if (M) std::cerr << "' on Module '" << M->getModuleIdentifier() << "'\n";
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, Function *F) {
  if (PassDebugging >= Executions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
              << P->getPassName();
    if (F) std::cerr << "' on Function '" << F->getName();
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintPassInformation(unsigned Depth, const char *Action,
                                   Pass *P, BasicBlock *BB) {
  if (PassDebugging >= Executions) {
    std::cerr << (void*)P << std::string(Depth*2+1, ' ') << Action << " '"
              << P->getPassName();
    if (BB) std::cerr << "' on BasicBlock '" << BB->getName();
    std::cerr << "'...\n";
  }
}

void PMDebug::PrintAnalysisSetInfo(unsigned Depth, const char *Msg,
                                   Pass *P, const std::vector<AnalysisID> &Set){
  if (PassDebugging >= Details && !Set.empty()) {
    std::cerr << (void*)P << std::string(Depth*2+3, ' ') << Msg << " Analyses:";
    for (unsigned i = 0; i != Set.size(); ++i) {
      if (i) std::cerr << ",";
      std::cerr << " " << Set[i]->getPassName();
    }
    std::cerr << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//

void ModulePass::addToPassManager(ModulePassManager *PM, AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

bool Pass::mustPreserveAnalysisID(const PassInfo *AnalysisID) const {
  return Resolver->getAnalysisToUpdate(AnalysisID) != 0;
}

// dumpPassStructure - Implement the -debug-passes=Structure option
void Pass::dumpPassStructure(unsigned Offset) {
  std::cerr << std::string(Offset*2, ' ') << getPassName() << "\n";
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

// dump - call print(std::cerr);
void Pass::dump() const {
  print(std::cerr, 0);
}

//===----------------------------------------------------------------------===//
// ImmutablePass Implementation
//
void ImmutablePass::addToPassManager(ModulePassManager *PM, 
                                     AnalysisUsage &AU) {
  PM->addPass(this, AU);
}


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

void FunctionPass::addToPassManager(ModulePassManager *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void FunctionPass::addToPassManager(FunctionPassManagerT *PM,
                                    AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

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

void BasicBlockPass::addToPassManager(FunctionPassManagerT *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}

void BasicBlockPass::addToPassManager(BasicBlockPassManager *PM,
                                      AnalysisUsage &AU) {
  PM->addPass(this, AU);
}


//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//
static std::map<TypeInfo, PassInfo*> *PassInfoMap = 0;
static std::vector<PassRegistrationListener*> *Listeners = 0;

// getPassInfo - Return the PassInfo data structure that corresponds to this
// pass...
const PassInfo *Pass::getPassInfo() const {
  if (PassInfoCache) return PassInfoCache;
  return lookupPassInfo(typeid(*this));
}

const PassInfo *Pass::lookupPassInfo(const std::type_info &TI) {
  if (PassInfoMap == 0) return 0;
  std::map<TypeInfo, PassInfo*>::iterator I = PassInfoMap->find(TI);
  return (I != PassInfoMap->end()) ? I->second : 0;
}

void RegisterPassBase::registerPass() {
  if (PassInfoMap == 0)
    PassInfoMap = new std::map<TypeInfo, PassInfo*>();

  assert(PassInfoMap->find(PIObj.getTypeInfo()) == PassInfoMap->end() &&
         "Pass already registered!");
  PassInfoMap->insert(std::make_pair(TypeInfo(PIObj.getTypeInfo()), &PIObj));

  // Notify any listeners...
  if (Listeners)
    for (std::vector<PassRegistrationListener*>::iterator
           I = Listeners->begin(), E = Listeners->end(); I != E; ++I)
      (*I)->passRegistered(&PIObj);
}

void RegisterPassBase::unregisterPass() {
  assert(PassInfoMap && "Pass registered but not in map!");
  std::map<TypeInfo, PassInfo*>::iterator I =
    PassInfoMap->find(PIObj.getTypeInfo());
  assert(I != PassInfoMap->end() && "Pass registered but not in map!");

  // Remove pass from the map...
  PassInfoMap->erase(I);
  if (PassInfoMap->empty()) {
    delete PassInfoMap;
    PassInfoMap = 0;
  }

  // Notify any listeners...
  if (Listeners)
    for (std::vector<PassRegistrationListener*>::iterator
           I = Listeners->begin(), E = Listeners->end(); I != E; ++I)
      (*I)->passUnregistered(&PIObj);
}

//===----------------------------------------------------------------------===//
//                  Analysis Group Implementation Code
//===----------------------------------------------------------------------===//

struct AnalysisGroupInfo {
  const PassInfo *DefaultImpl;
  std::set<const PassInfo *> Implementations;
  AnalysisGroupInfo() : DefaultImpl(0) {}
};

static std::map<const PassInfo *, AnalysisGroupInfo> *AnalysisGroupInfoMap = 0;

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

    // Lazily allocate to avoid nasty initialization order dependencies
    if (AnalysisGroupInfoMap == 0)
      AnalysisGroupInfoMap = new std::map<const PassInfo *,AnalysisGroupInfo>();

    AnalysisGroupInfo &AGI = (*AnalysisGroupInfoMap)[InterfaceInfo];
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
}

void RegisterAGBase::setGroupName(const char *Name) {
  assert(InterfaceInfo->getPassName()[0] == 0 && "Interface Name already set!");
  InterfaceInfo->setPassName(Name);
}

RegisterAGBase::~RegisterAGBase() {
  if (ImplementationInfo) {
    assert(AnalysisGroupInfoMap && "Inserted into map, but map doesn't exist?");
    AnalysisGroupInfo &AGI = (*AnalysisGroupInfoMap)[InterfaceInfo];

    assert(AGI.Implementations.count(ImplementationInfo) &&
           "Pass not a member of analysis group?");

    if (AGI.DefaultImpl == ImplementationInfo)
      AGI.DefaultImpl = 0;

    AGI.Implementations.erase(ImplementationInfo);

    // Last member of this analysis group? Unregister PassInfo, delete map entry
    if (AGI.Implementations.empty()) {
      assert(AGI.DefaultImpl == 0 &&
             "Default implementation didn't unregister?");
      AnalysisGroupInfoMap->erase(InterfaceInfo);
      if (AnalysisGroupInfoMap->empty()) {  // Delete map if empty
        delete AnalysisGroupInfoMap;
        AnalysisGroupInfoMap = 0;
      }
    }
  }
  
  if (InterfaceInfo == &PIObj)
    unregisterPass();
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
  if (PassInfoMap)
    for (std::map<TypeInfo, PassInfo*>::iterator I = PassInfoMap->begin(),
           E = PassInfoMap->end(); I != E; ++I)
      passEnumerate(I->second);
}

