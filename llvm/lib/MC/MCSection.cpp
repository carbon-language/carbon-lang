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

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) const {
  if (!End)
    End = Ctx.createTempSymbol("sec_end", true);
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

MCSection::~MCSection() {
}

