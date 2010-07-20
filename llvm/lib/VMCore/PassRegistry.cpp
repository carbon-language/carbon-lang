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
#include "llvm/System/Mutex.h"

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

void PassRegistry::registerPass(const PassInfo &PI) {
  sys::SmartScopedLock<true> Guard(Lock);
  bool Inserted =
    PassInfoMap.insert(std::make_pair(PI.getTypeInfo(),&PI)).second;
  assert(Inserted && "Pass registered multiple times!"); Inserted=Inserted;
  PassInfoStringMap[PI.getPassArgument()] = &PI;
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
void PassRegistry::registerAnalysisGroup(PassInfo *InterfaceInfo,
                                         const PassInfo *ImplementationInfo,
                                         bool isDefault) {
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
