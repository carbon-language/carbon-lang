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
// and registration of passes.  At application startup, passes are registered
// with the PassRegistry, which is later provided to the PassManager for 
// dependency resolution and similar tasks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSREGISTRY_H
#define LLVM_PASSREGISTRY_H

#include "llvm/ADT/StringMap.h"
#include "llvm/System/DataTypes.h"

namespace llvm {

class PassInfo;
struct PassRegistrationListener;

/// PassRegistry - This class manages the registration and intitialization of
/// the pass subsystem as application startup, and assists the PassManager
/// in resolving pass dependencies.
/// NOTE: PassRegistry is NOT thread-safe.  If you want to use LLVM on multiple
/// threads simultaneously, you will need to use a separate PassRegistry on
/// each thread.
class PassRegistry {
  mutable void *pImpl;
  void *getImpl() const;
   
public:
  static PassRegistry *getPassRegistry();
  
  const PassInfo *getPassInfo(const void *TI) const;
  const PassInfo *getPassInfo(StringRef Arg) const;
  
  void registerPass(const PassInfo &PI);
  void unregisterPass(const PassInfo &PI);
  
  /// Analysis Group Mechanisms.
  void registerAnalysisGroup(const void *InterfaceID, const void *PassID,
                             PassInfo& Registeree, bool isDefault);
  
  void enumerateWith(PassRegistrationListener *L);
  void addRegistrationListener(PassRegistrationListener *L);
  void removeRegistrationListener(PassRegistrationListener *L);
};

}

#endif
