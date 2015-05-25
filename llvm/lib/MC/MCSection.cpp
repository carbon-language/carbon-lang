//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MCSection
//===----------------------------------------------------------------------===//

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
