//===- MCSectionXCore.cpp - XCore-specific section representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MCSectionXCore class.
//
//===----------------------------------------------------------------------===//

#include "MCSectionXCore.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSectionXCore *
MCSectionXCore::Create(const StringRef &Section, unsigned Type,
                       unsigned Flags, SectionKind K,
                       bool isExplicit, MCContext &Ctx) {
  return new (Ctx) MCSectionXCore(Section, Type, Flags, K, isExplicit);
}


/// PrintTargetSpecificSectionFlags - This handles the XCore-specific cp/dp
/// section flags.
void MCSectionXCore::PrintTargetSpecificSectionFlags(const MCAsmInfo &TAI,
                                                     raw_ostream &OS) const {
  if (getFlags() & MCSectionXCore::SHF_CP_SECTION)
    OS << 'c';
  if (getFlags() & MCSectionXCore::SHF_DP_SECTION)
    OS << 'd';
}
