//===--- TargetRegistry.cpp - Target registration -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetRegistry.h"
#include "llvm/System/Host.h"
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
  // Provide special warning when no targets are initialized.
  if (begin() == end()) {
    Error = "Unable to find target for this triple (no targets are registered)";
    return 0;
  }
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
    Error = "No available targets are compatible with this triple";
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
  // Provide special warning when no targets are initialized.
  if (begin() == end()) {
    Error = "Unable to find target for this module (no targets are registered)";
    return 0;
  }

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

  // FIXME: This is a hack to ignore super weak matches like msil, etc. and look
  // by host instead. They will be found again via the triple.
  if (Best && BestQuality == 1)    
    Best = EquallyBest = 0;

  // If that failed, try looking up the host triple.
  if (!Best)
    Best = getClosestStaticTargetForTriple(sys::getHostTriple(), Error);

  if (!Best) {
    Error = "No available targets are compatible with this module";
    return Best;
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
  std::string Triple = sys::getHostTriple();

  // Provide special warning when no targets are initialized.
  if (begin() == end()) {
    Error = "No JIT is available for this host (no targets are registered)";
    return 0;
  }

  const Target *Best = 0, *EquallyBest = 0;
  unsigned BestQuality = 0;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (!it->hasJIT())
      continue;

    if (unsigned Qual = it->TripleMatchQualityFn(Triple)) {
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
                                    bool HasJIT) {
  assert(Name && ShortDesc && TQualityFn && MQualityFn &&
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
  T.HasJIT = HasJIT;
}

