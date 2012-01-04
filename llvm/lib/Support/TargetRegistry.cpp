//===--- TargetRegistry.cpp - Target registration -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>
using namespace llvm;

// Clients are responsible for avoid race conditions in registration.
static Target *FirstTarget = 0;

TargetRegistry::iterator TargetRegistry::begin() {
  return iterator(FirstTarget);
}

const Target *TargetRegistry::lookupTarget(const std::string &TT,
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
    Error = "No available targets are compatible with this triple, "
      "see -version for the available targets.";
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

void TargetRegistry::RegisterTarget(Target &T,
                                    const char *Name,
                                    const char *ShortDesc,
                                    Target::TripleMatchQualityFnTy TQualityFn,
                                    bool HasJIT) {
  assert(Name && ShortDesc && TQualityFn &&
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
  T.HasJIT = HasJIT;
}

const Target *TargetRegistry::getClosestTargetForJIT(std::string &Error) {
  const Target *TheTarget = lookupTarget(sys::getHostTriple(), Error);

  if (TheTarget && !TheTarget->hasJIT()) {
    Error = "No JIT compatible target available for this host";
    return 0;
  }

  return TheTarget;
}

static int TargetArraySortFn(const void *LHS, const void *RHS) {
  typedef std::pair<StringRef, const Target*> pair_ty;
  return ((const pair_ty*)LHS)->first.compare(((const pair_ty*)RHS)->first);
}

void TargetRegistry::printRegisteredTargetsForVersion() {
  std::vector<std::pair<StringRef, const Target*> > Targets;
  size_t Width = 0;
  for (TargetRegistry::iterator I = TargetRegistry::begin(),
       E = TargetRegistry::end();
       I != E; ++I) {
    Targets.push_back(std::make_pair(I->getName(), &*I));
    Width = std::max(Width, Targets.back().first.size());
  }
  array_pod_sort(Targets.begin(), Targets.end(), TargetArraySortFn);

  raw_ostream &OS = outs();
  OS << "  Registered Targets:\n";
  for (unsigned i = 0, e = Targets.size(); i != e; ++i) {
    OS << "    " << Targets[i].first;
    OS.indent(Width - Targets[i].first.size()) << " - "
      << Targets[i].second->getShortDescription() << '\n';
  }
  if (Targets.empty())
    OS << "    (none)\n";
}
