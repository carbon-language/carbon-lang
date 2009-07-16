//===--- TargetRegistry.cpp - Target registration -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetRegistry.h"
#include <cassert>
using namespace llvm;

// Clients are responsible for avoid race conditions in registration.
static Target *FirstTarget = 0;

TargetRegistry::iterator TargetRegistry::begin() {
  return iterator(FirstTarget);
}

const Target *
TargetRegistry::getClosestStaticTargetForTriple(const std::string &TT,
                                                std::string &Error) {
  const Target *Best = 0, *EquallyBest = 0;
  unsigned BestQuality = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (unsigned Qual = it->TripleMatchQualityFn(TT)) {
      if (!Best || Qual > BestQuality) {
        Best = &*it;
        EquallyBest = 0;
        BestQuality = Qual;
      } else if (Qual == BestQuality)
        EquallyBest = &*it;
    }
  }

  if (!Best) {
    Error = "No available targets are compatible with this module";
    return 0;
  }

  // Otherwise, take the best target, but make sure we don't have two equally
  // good best targets.
  if (EquallyBest) {
    Error = std::string("Cannot choose between targets \"") +
      Best->Name  + "\" and \"" + EquallyBest->Name + "\"";
    return 0;
  }

  return Best;
}

const Target *
TargetRegistry::getClosestStaticTargetForModule(const Module &M,
                                                std::string &Error) {
  const Target *Best = 0, *EquallyBest = 0;
  unsigned BestQuality = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (unsigned Qual = it->ModuleMatchQualityFn(M)) {
      if (!Best || Qual > BestQuality) {
        Best = &*it;
        EquallyBest = 0;
        BestQuality = Qual;
      } else if (Qual == BestQuality)
        EquallyBest = &*it;
    }
  }

  if (!Best) {
    Error = "No available targets are compatible with this module";
    return 0;
  }

  // Otherwise, take the best target, but make sure we don't have two equally
  // good best targets.
  if (EquallyBest) {
    Error = std::string("Cannot choose between targets \"") +
      Best->Name  + "\" and \"" + EquallyBest->Name + "\"";
    return 0;
  }

  return Best;
}

const Target *
TargetRegistry::getClosestTargetForJIT(std::string &Error) {
  const Target *Best = 0, *EquallyBest = 0;
  unsigned BestQuality = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (unsigned Qual = it->JITMatchQualityFn()) {
      if (!Best || Qual > BestQuality) {
        Best = &*it;
        EquallyBest = 0;
        BestQuality = Qual;
      } else if (Qual == BestQuality)
        EquallyBest = &*it;
    }
  }

  if (!Best) {
    Error = "No JIT is available for this host";
    return 0;
  }

  // Return the best, ignoring ties.
  return Best;
}

void TargetRegistry::RegisterTarget(Target &T,
                                    const char *Name,
                                    const char *ShortDesc,
                                    Target::TripleMatchQualityFnTy TQualityFn,
                                    Target::ModuleMatchQualityFnTy MQualityFn,
                                    Target::JITMatchQualityFnTy JITQualityFn) {
  assert(Name && ShortDesc && TQualityFn && MQualityFn && JITQualityFn &&
         "Missing required target information!");

  // Check if this target has already been initialized, we allow this as a
  // convenience to some clients.
  if (T.Name)
    return;
         
  // Add to the list of targets.
  T.Next = FirstTarget;
  FirstTarget = &T;

  T.Name = Name;
  T.ShortDesc = ShortDesc;
  T.TripleMatchQualityFn = TQualityFn;
  T.ModuleMatchQualityFn = MQualityFn;
  T.JITMatchQualityFn = JITQualityFn;
}

