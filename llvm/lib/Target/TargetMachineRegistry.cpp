//===-- TargetMachineRegistry.cpp - Target Auto Registration Impl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

/// List - This is the main list of all of the registered target machines.
const TargetMachineRegistry::Entry *TargetMachineRegistry::List = 0;

/// Listeners - All of the listeners registered to get notified when new targets
/// are loaded.
static TargetRegistrationListener *Listeners = 0;

TargetMachineRegistry::Entry::Entry(const char *N, const char *SD,
                       TargetMachine *(*CF)(const Module &, IntrinsicLowering*,
                                            const std::string &),
                           unsigned (*MMF)(const Module &M), unsigned (*JMF)())
  : Name(N), ShortDesc(SD), CtorFn(CF), ModuleMatchQualityFn(MMF),
    JITMatchQualityFn(JMF), Next(List) {
  List = this;
  for (TargetRegistrationListener *L = Listeners; L; L = L->getNext())
    L->targetRegistered(this);
}

TargetRegistrationListener::TargetRegistrationListener() {
  Next = Listeners;
  if (Next) Next->Prev = &Next;
  Prev = &Listeners;
  Listeners = this;
}

TargetRegistrationListener::~TargetRegistrationListener() {
  *Prev = Next;
}

/// getClosestStaticTargetForModule - Given an LLVM module, pick the best target
/// that is compatible with the module.  If no close target can be found, this
/// returns null and sets the Error string to a reason.
const TargetMachineRegistry::Entry *
TargetMachineRegistry::getClosestStaticTargetForModule(const Module &M,
                                                       std::string &Error) {
  std::vector<std::pair<unsigned, const Entry *> > UsableTargets;
  for (const Entry *E = getList(); E; E = E->getNext())
    if (unsigned Qual = E->ModuleMatchQualityFn(M))
      UsableTargets.push_back(std::make_pair(Qual, E));

  if (UsableTargets.empty()) {
    Error = "No available targets are compatible with this module";
    return 0;
  } else if (UsableTargets.size() == 1)
    return UsableTargets.back().second;

  // Otherwise, take the best target, but make sure we don't have to equally
  // good best targets.
  std::sort(UsableTargets.begin(), UsableTargets.end());
  if (UsableTargets.back().first ==UsableTargets[UsableTargets.size()-2].first){
    Error = "Cannot choose between targets \"" +
      std::string(UsableTargets.back().second->Name) + "\" and \"" +
      std::string(UsableTargets[UsableTargets.size()-2].second->Name) + "\"";
    return 0;
  }
  return UsableTargets.back().second;
}

/// getClosestTargetForJIT - Given an LLVM module, pick the best target that
/// is compatible with the current host and the specified module.  If no
/// close target can be found, this returns null and sets the Error string
/// to a reason.
const TargetMachineRegistry::Entry *
TargetMachineRegistry::getClosestTargetForJIT(std::string &Error) {
  std::vector<std::pair<unsigned, const Entry *> > UsableTargets;
  for (const Entry *E = getList(); E; E = E->getNext())
    if (unsigned Qual = E->JITMatchQualityFn())
      UsableTargets.push_back(std::make_pair(Qual, E));

  if (UsableTargets.empty()) {
    Error = "No JIT is available for this host";
    return 0;
  } else if (UsableTargets.size() == 1)
    return UsableTargets.back().second;

  // Otherwise, take the best target.  If there is a tie, just pick one.
  unsigned MaxQual = UsableTargets.front().first;
  const Entry *MaxQualTarget = UsableTargets.front().second;

  for (unsigned i = 1, e = UsableTargets.size(); i != e; ++i)
    if (UsableTargets[i].first > MaxQual) {
      MaxQual = UsableTargets[i].first;
      MaxQualTarget = UsableTargets[i].second;
    }

  return MaxQualTarget;
}

