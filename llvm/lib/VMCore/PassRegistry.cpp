//===- PassRegistry.cpp - Pass Registration Implementation ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PassRegistry, with which passes are registered on
// initialization, and supports the PassManager in dependency resolution.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

static PassRegistry *PassRegistryObj = 0;
PassRegistry *PassRegistry::getPassRegistry() {
  // Use double-checked locking to safely initialize the registrar when
  // we're running in multithreaded mode.
  PassRegistry* tmp = PassRegistryObj;
  if (llvm_is_multithreaded()) {
    sys::MemoryFence();
    if (!tmp) {
      llvm_acquire_global_lock();
      tmp = PassRegistryObj;
      if (!tmp) {
        tmp = new PassRegistry();
        sys::MemoryFence();
        PassRegistryObj = tmp;
      }
      llvm_release_global_lock();
    }
  } else if (!tmp) {
    PassRegistryObj = new PassRegistry();
  }
  
  return PassRegistryObj;
}

namespace {

// FIXME: We use ManagedCleanup to erase the pass registrar on shutdown.
// Unfortunately, passes are registered with static ctors, and having
// llvm_shutdown clear this map prevents successful ressurection after 
// llvm_shutdown is run.  Ideally we should find a solution so that we don't
// leak the map, AND can still resurrect after shutdown.
void cleanupPassRegistry(void*) {
  if (PassRegistryObj) {
    delete PassRegistryObj;
    PassRegistryObj = 0;
  }
}
ManagedCleanup<&cleanupPassRegistry> registryCleanup ATTRIBUTE_USED;

}

const PassInfo *PassRegistry::getPassInfo(intptr_t TI) const {
  sys::SmartScopedLock<true> Guard(Lock);
  MapType::const_iterator I = PassInfoMap.find(TI);
  return I != PassInfoMap.end() ? I->second : 0;
}

const PassInfo *PassRegistry::getPassInfo(StringRef Arg) const {
  sys::SmartScopedLock<true> Guard(Lock);
  StringMapType::const_iterator I = PassInfoStringMap.find(Arg);
  return I != PassInfoStringMap.end() ? I->second : 0;
}

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//

void PassRegistry::registerPass(const PassInfo &PI) {
  sys::SmartScopedLock<true> Guard(Lock);
  bool Inserted =
    PassInfoMap.insert(std::make_pair(PI.getTypeInfo(),&PI)).second;
  assert(Inserted && "Pass registered multiple times!"); Inserted=Inserted;
  PassInfoStringMap[PI.getPassArgument()] = &PI;
  
  // Notify any listeners.
  for (std::vector<PassRegistrationListener*>::iterator
       I = Listeners.begin(), E = Listeners.end(); I != E; ++I)
    (*I)->passRegistered(&PI);
}

void PassRegistry::unregisterPass(const PassInfo &PI) {
  sys::SmartScopedLock<true> Guard(Lock);
  MapType::iterator I = PassInfoMap.find(PI.getTypeInfo());
  assert(I != PassInfoMap.end() && "Pass registered but not in map!");
  
  // Remove pass from the map.
  PassInfoMap.erase(I);
  PassInfoStringMap.erase(PI.getPassArgument());
}

void PassRegistry::enumerateWith(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(Lock);
  for (MapType::const_iterator I = PassInfoMap.begin(),
       E = PassInfoMap.end(); I != E; ++I)
    L->passEnumerate(I->second);
}


/// Analysis Group Mechanisms.
void PassRegistry::registerAnalysisGroup(intptr_t InterfaceID, 
                                         intptr_t PassID,
                                         PassInfo& Registeree,
                                         bool isDefault) {
  PassInfo *InterfaceInfo =  const_cast<PassInfo*>(getPassInfo(InterfaceID));
  if (InterfaceInfo == 0) {
    // First reference to Interface, register it now.
    registerPass(Registeree);
    InterfaceInfo = &Registeree;
  }
  assert(Registeree.isAnalysisGroup() && 
         "Trying to join an analysis group that is a normal pass!");

  if (PassID) {
    PassInfo *ImplementationInfo = const_cast<PassInfo*>(getPassInfo(PassID));
    assert(ImplementationInfo &&
           "Must register pass before adding to AnalysisGroup!");

    // Make sure we keep track of the fact that the implementation implements
    // the interface.
    ImplementationInfo->addInterfaceImplemented(InterfaceInfo);

    sys::SmartScopedLock<true> Guard(Lock);
    AnalysisGroupInfo &AGI = AnalysisGroupInfoMap[InterfaceInfo];
    assert(AGI.Implementations.count(ImplementationInfo) == 0 &&
           "Cannot add a pass to the same analysis group more than once!");
    AGI.Implementations.insert(ImplementationInfo);
    if (isDefault) {
      assert(InterfaceInfo->getNormalCtor() == 0 &&
             "Default implementation for analysis group already specified!");
      assert(ImplementationInfo->getNormalCtor() &&
           "Cannot specify pass as default if it does not have a default ctor");
      InterfaceInfo->setNormalCtor(ImplementationInfo->getNormalCtor());
    }
  }
}

void PassRegistry::addRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(Lock);
  Listeners.push_back(L);
}

void PassRegistry::removeRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(Lock);
  std::vector<PassRegistrationListener*>::iterator I =
    std::find(Listeners.begin(), Listeners.end(), L);
  assert(I != Listeners.end() && "PassRegistrationListener not registered!");
  Listeners.erase(I);
}
