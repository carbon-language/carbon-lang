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
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TypeInfo.h"
#include <algorithm>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Pass Implementation
//

// Force out-of-line virtual method.
Pass::~Pass() { 
  delete Resolver; 
}

// Force out-of-line virtual method.
ModulePass::~ModulePass() { }

bool Pass::mustPreserveAnalysisID(const PassInfo *AnalysisID) const {
  return Resolver->getAnalysisToUpdate(AnalysisID, true) != 0;
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
// Force out-of-line virtual method.
ImmutablePass::~ImmutablePass() { }

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

// run - On a module, we run this pass by initializing, runOnFunction'ing once
// for every function in the module, then by finalizing.
//
bool FunctionPass::runOnModule(Module &M) {
  bool Changed = doInitialization(M);

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration())      // Passes are not run on external functions!
    Changed |= runOnFunction(*I);

  return Changed | doFinalization(M);
}

// run - On a function, we simply initialize, run the function, then finalize.
//
bool FunctionPass::run(Function &F) {
  if (F.isDeclaration()) return false;// Passes are not run on external functions!

  bool Changed = doInitialization(*F.getParent());
  Changed |= runOnFunction(F);
  return Changed | doFinalization(*F.getParent());
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

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//
namespace {
class PassRegistrar {
  /// PassInfoMap - Keep track of the passinfo object for each registered llvm
  /// pass.
  std::map<intptr_t, PassInfo*> PassInfoMap;
  
  /// AnalysisGroupInfo - Keep track of information for each analysis group.
  struct AnalysisGroupInfo {
    const PassInfo *DefaultImpl;
    std::set<const PassInfo *> Implementations;
    AnalysisGroupInfo() : DefaultImpl(0) {}
  };
  
  /// AnalysisGroupInfoMap - Information for each analysis group.
  std::map<const PassInfo *, AnalysisGroupInfo> AnalysisGroupInfoMap;

public:
  
  const PassInfo *GetPassInfo(intptr_t TI) const {
    std::map<intptr_t, PassInfo*>::const_iterator I = PassInfoMap.find(TI);
    return I != PassInfoMap.end() ? I->second : 0;
  }
  
  void RegisterPass(PassInfo &PI) {
    bool Inserted =
      PassInfoMap.insert(std::make_pair(PI.getTypeInfo(),&PI)).second;
    assert(Inserted && "Pass registered multiple times!");
  }
  
  void UnregisterPass(PassInfo &PI) {
    std::map<intptr_t, PassInfo*>::iterator I =
      PassInfoMap.find(PI.getTypeInfo());
    assert(I != PassInfoMap.end() && "Pass registered but not in map!");
    
    // Remove pass from the map.
    PassInfoMap.erase(I);
  }
  
  void EnumerateWith(PassRegistrationListener *L) {
    for (std::map<intptr_t, PassInfo*>::const_iterator I = PassInfoMap.begin(),
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

static std::vector<PassRegistrationListener*> *Listeners = 0;

// FIXME: This should use ManagedStatic to manage the pass registrar.
// Unfortunately, we can't do this, because passes are registered with static
// ctors, and having llvm_shutdown clear this map prevents successful
// ressurection after llvm_shutdown is run.
static PassRegistrar *getPassRegistrar() {
  static PassRegistrar *PassRegistrarObj = 0;
  if (!PassRegistrarObj)
    PassRegistrarObj = new PassRegistrar();
  return PassRegistrarObj;
}

// getPassInfo - Return the PassInfo data structure that corresponds to this
// pass...
const PassInfo *Pass::getPassInfo() const {
  return lookupPassInfo(PassID);
}

const PassInfo *Pass::lookupPassInfo(intptr_t TI) {
  return getPassRegistrar()->GetPassInfo(TI);
}

void RegisterPassBase::registerPass() {
  getPassRegistrar()->RegisterPass(PIObj);

  // Notify any listeners.
  if (Listeners)
    for (std::vector<PassRegistrationListener*>::iterator
           I = Listeners->begin(), E = Listeners->end(); I != E; ++I)
      (*I)->passRegistered(&PIObj);
}

void RegisterPassBase::unregisterPass() {
  getPassRegistrar()->UnregisterPass(PIObj);
}

//===----------------------------------------------------------------------===//
//                  Analysis Group Implementation Code
//===----------------------------------------------------------------------===//

// RegisterAGBase implementation
//
RegisterAGBase::RegisterAGBase(intptr_t InterfaceID,
                               intptr_t PassID, bool isDefault)
  : RegisterPassBase(InterfaceID),
    ImplementationInfo(0), isDefaultImplementation(isDefault) {

  InterfaceInfo = const_cast<PassInfo*>(Pass::lookupPassInfo(InterfaceID));
  if (InterfaceInfo == 0) {
    // First reference to Interface, register it now.
    registerPass();
    InterfaceInfo = &PIObj;
  }
  assert(PIObj.isAnalysisGroup() &&
         "Trying to join an analysis group that is a normal pass!");

  if (PassID) {
    ImplementationInfo = Pass::lookupPassInfo(PassID);
    assert(ImplementationInfo &&
           "Must register pass before adding to AnalysisGroup!");

    // Make sure we keep track of the fact that the implementation implements
    // the interface.
    PassInfo *IIPI = const_cast<PassInfo*>(ImplementationInfo);
    IIPI->addInterfaceImplemented(InterfaceInfo);
    
    getPassRegistrar()->RegisterAnalysisGroup(InterfaceInfo, IIPI, isDefault);
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
  getPassRegistrar()->EnumerateWith(this);
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


