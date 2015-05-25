//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MCSection
//===----------------------------------------------------------------------===//

MCSection::MCSection(SectionVariant V, SectionKind K, MCSymbol *Begin)
    : Begin(Begin), HasInstructions(false), Data(*this), Variant(V), Kind(K) {}

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) {
  if (!End)
    End = Ctx.createTempSymbol("sec_end", true);
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

MCSection::~MCSection() {
}

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

MCSectionData::iterator MCSection::begin() { return Data.begin(); }

MCSectionData::iterator MCSection::end() { return Data.end(); }

MCSectionData::FragmentListType &MCSection::getFragmentList() {
  return Data.getFragmentList();
}

MCSectionData::iterator MCSectionData::begin() { return Fragments.begin(); }

MCSectionData::iterator MCSectionData::end() { return Fragments.end(); }

MCSectionData::reverse_iterator MCSectionData::rbegin() {
  return Fragments.rbegin();
}

MCSectionData::reverse_iterator MCSectionData::rend() {
  return Fragments.rend();
}

size_t MCSectionData::size() const { return Fragments.size(); }

bool MCSectionData::empty() const { return Fragments.empty(); }
