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
#include "llvm/MC/MCAsmInfo.h"
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
Create(StringRef Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  char *NameCopy = static_cast<char*>(
    Ctx.Allocate(Name.size(), /*Alignment=*/1));
  memcpy(NameCopy, Name.data(), Name.size());
  return new (Ctx) MCSectionCOFF(StringRef(NameCopy, Name.size()),
                                 IsDirective, K);
}

void MCSectionCOFF::PrintSwitchToSection(const MCAsmInfo &MAI,
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
