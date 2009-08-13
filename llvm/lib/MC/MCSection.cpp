//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MCSection
//===----------------------------------------------------------------------===//

MCSection::~MCSection() {
}

//===----------------------------------------------------------------------===//
// MCSectionCOFF
//===----------------------------------------------------------------------===//

MCSectionCOFF *MCSectionCOFF::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionCOFF(Name, IsDirective, K);
}

void MCSectionCOFF::PrintSwitchToSection(const TargetAsmInfo &TAI,
                                         raw_ostream &OS) const {
  
  if (isDirective()) {
    OS << getName() << '\n';
    return;
  }
  OS << "\t.section\t" << getName() << ",\"";
  if (getKind().isText())
    OS << 'x';
  if (getKind().isWriteable())
    OS << 'w';
  OS << "\"\n";
}
