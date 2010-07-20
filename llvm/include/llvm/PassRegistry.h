//===- llvm/PassRegistry.h - Pass Information Registry ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines PassRegistry, a class that is used in the initialization
// and registration of passes.  At initialization, passes are registered with
// the PassRegistry, which is later provided to the PassManager for dependency
// resolution and similar tasks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSREGISTRY_H
#define LLVM_PASSREGISTRY_H

#include "llvm/PassSupport.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/System/DataTypes.h"
#include "llvm/System/Mutex.h"
#include <map>
#include <set>

using namespace llvm;

namespace llvm {

class PassRegistry {
  /// Guards the contents of this class.
  mutable sys::SmartMutex<true> Lock;
  
  /// PassInfoMap - Keep track of the PassInfo object for each registered pass.
  typedef std::map<intptr_t, const PassInfo*> MapType;
  MapType PassInfoMap;
  
  typedef StringMap<const PassInfo*> StringMapType;
  StringMapType PassInfoStringMap;
  
  /// AnalysisGroupInfo - Keep track of information for each analysis group.
  struct AnalysisGroupInfo {
    std::set<const PassInfo *> Implementations;
  };
  std::map<const PassInfo*, AnalysisGroupInfo> AnalysisGroupInfoMap;

public:
  const PassInfo *getPassInfo(intptr_t TI) const;
  const PassInfo *getPassInfo(StringRef Arg) const;
  
  void registerPass(const PassInfo &PI);
  void unregisterPass(const PassInfo &PI);
  
  /// Analysis Group Mechanisms.
  void registerAnalysisGroup(PassInfo *InterfaceInfo,
                             const PassInfo *ImplementationInfo,
                             bool isDefault);
  
  void enumerateWith(PassRegistrationListener *L);
};

}

#endif
