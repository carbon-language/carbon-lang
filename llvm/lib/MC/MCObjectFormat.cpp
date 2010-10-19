//===- lib/MC/MCObjectFormat.cpp - MCObjectFormat implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCSymbol.h"

using namespace llvm;

MCObjectFormat::~MCObjectFormat() {
}

bool MCELFObjectFormat::isAbsolute(bool IsSet, const MCSymbol &A,
                                   const MCSymbol &B) const {
  // On ELF A - B is absolute if A and B are in the same section.
  return &A.getSection() == &B.getSection();
}

bool MCMachOObjectFormat::isAbsolute(bool IsSet, const MCSymbol &A,
                                     const MCSymbol &B) const  {
  // On MachO A - B is absolute only if in a set.
  return IsSet;
}

bool MCCOFFObjectFormat::isAbsolute(bool IsSet, const MCSymbol &A,
                                    const MCSymbol &B) const  {
  // On COFF A - B is absolute if A and B are in the same section.
  return &A.getSection() == &B.getSection();
}
