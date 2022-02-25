//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

MCSection::MCSection(SectionVariant V, StringRef Name, SectionKind K,
                     MCSymbol *Begin)
    : Begin(Begin), BundleGroupBeforeFirstInst(false), HasInstructions(false),
      IsRegistered(false), DummyFragment(this), Name(Name), Variant(V),
      Kind(K) {}

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) {
  if (!End)
    End = Ctx.createTempSymbol("sec_end");
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

MCSection::~MCSection() = default;

void MCSection::setBundleLockState(BundleLockStateType NewState) {
  if (NewState == NotBundleLocked) {
    if (BundleLockNestingDepth == 0) {
      report_fatal_error("Mismatched bundle_lock/unlock directives");
    }
    if (--BundleLockNestingDepth == 0) {
      BundleLockState = NotBundleLocked;
    }
    return;
  }

  // If any of the directives is an align_to_end directive, the whole nested
  // group is align_to_end. So don't downgrade from align_to_end to just locked.
  if (BundleLockState != BundleLockedAlignToEnd) {
    BundleLockState = NewState;
  }
  ++BundleLockNestingDepth;
}

MCSection::iterator
MCSection::getSubsectionInsertionPoint(unsigned Subsection) {
  if (Subsection == 0 && SubsectionFragmentMap.empty())
    return end();

  SmallVectorImpl<std::pair<unsigned, MCFragment *>>::iterator MI = lower_bound(
      SubsectionFragmentMap, std::make_pair(Subsection, (MCFragment *)nullptr));
  bool ExactMatch = false;
  if (MI != SubsectionFragmentMap.end()) {
    ExactMatch = MI->first == Subsection;
    if (ExactMatch)
      ++MI;
  }
  iterator IP;
  if (MI == SubsectionFragmentMap.end())
    IP = end();
  else
    IP = MI->second->getIterator();
  if (!ExactMatch && Subsection != 0) {
    // The GNU as documentation claims that subsections have an alignment of 4,
    // although this appears not to be the case.
    MCFragment *F = new MCDataFragment();
    SubsectionFragmentMap.insert(MI, std::make_pair(Subsection, F));
    getFragmentList().insert(IP, F);
    F->setParent(this);
    F->setSubsectionNumber(Subsection);
  }

  return IP;
}

StringRef MCSection::getVirtualSectionKind() const { return "virtual"; }

void MCSection::addPendingLabel(MCSymbol *label, unsigned Subsection) {
  PendingLabels.push_back(PendingLabel(label, Subsection));
}

void MCSection::flushPendingLabels(MCFragment *F, uint64_t FOffset,
				   unsigned Subsection) {
  if (PendingLabels.empty())
    return;

  // Set the fragment and fragment offset for all pending symbols in the
  // specified Subsection, and remove those symbols from the pending list.
  for (auto It = PendingLabels.begin(); It != PendingLabels.end(); ++It) {
    PendingLabel& Label = *It;
    if (Label.Subsection == Subsection) {
      Label.Sym->setFragment(F);
      Label.Sym->setOffset(FOffset);
      PendingLabels.erase(It--);
    }
  }
}

void MCSection::flushPendingLabels() {
  // Make sure all remaining pending labels point to data fragments, by
  // creating new empty data fragments for each Subsection with labels pending.
  while (!PendingLabels.empty()) {
    PendingLabel& Label = PendingLabels[0];
    iterator CurInsertionPoint =
      this->getSubsectionInsertionPoint(Label.Subsection);
    MCFragment *F = new MCDataFragment();
    getFragmentList().insert(CurInsertionPoint, F);
    F->setParent(this);
    flushPendingLabels(F, 0, Label.Subsection);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCSection::dump() const {
  raw_ostream &OS = errs();

  OS << "<MCSection";
  OS << " Fragments:[\n      ";
  for (auto it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin())
      OS << ",\n      ";
    it->dump();
  }
  OS << "]>";
}
#endif
