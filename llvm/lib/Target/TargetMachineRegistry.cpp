//===-- TargetMachineRegistry.cpp - Target Auto Registration Impl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the RegisterTarget class, which TargetMachine
// implementations should use to register themselves with the system.  This file
// also exposes the TargetMachineRegistry class, which allows tools to inspect
// all of registered targets.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachineRegistry.h"
#include <algorithm>
using namespace llvm;

/// getClosestStaticTargetForModule - Given an LLVM module, pick the best target
/// that is compatible with the module.  If no close target can be found, this
/// returns null and sets the Error string to a reason.
const TargetMachineRegistry::entry *
TargetMachineRegistry::getClosestStaticTargetForModule(const Module &M,
                                                       std::string &Error) {
  const Target *T = TargetRegistry::getClosestStaticTargetForModule(M, Error);
  if (!T)
    return 0;
  // FIXME: Temporary hack, please remove.
  return new TargetMachineRegistry::entry(T->Name, T->ShortDesc,
                                          T->TargetMachineCtorFn,
                                          T->ModuleMatchQualityFn,
                                          T->JITMatchQualityFn); 
}

/// getClosestTargetForJIT - Pick the best target that is compatible with
/// the current host.  If no close target can be found, this returns null
/// and sets the Error string to a reason.
const TargetMachineRegistry::entry *
TargetMachineRegistry::getClosestTargetForJIT(std::string &Error) {
  const Target *T = TargetRegistry::getClosestTargetForJIT(Error);
  if (!T)
    return 0;
  // FIXME: Temporary hack, please remove.
  return new TargetMachineRegistry::entry(T->Name, T->ShortDesc,
                                          T->TargetMachineCtorFn,
                                          T->ModuleMatchQualityFn,
                                          T->JITMatchQualityFn); 
}

