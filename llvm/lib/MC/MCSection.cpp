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
using namespace llvm;

MCSection::~MCSection() {
}

MCSection::MCSection(const StringRef &N, bool isDirective, SectionKind K, 
                     MCContext &Ctx)
  : Name(N), IsDirective(isDirective), Kind(K) {
  MCSection *&Entry = Ctx.Sections[Name];
  assert(Entry == 0 && "Multiple sections with the same name created");
  Entry = this;
}

MCSection *MCSection::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSection(Name, IsDirective, K, Ctx);
}


MCSectionELF *MCSectionELF::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionELF(Name, IsDirective, K, Ctx);
}


MCSectionMachO *MCSectionMachO::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionMachO(Name, IsDirective, K, Ctx);
}


MCSectionPECOFF *MCSectionPECOFF::
Create(const StringRef &Name, bool IsDirective, SectionKind K, MCContext &Ctx) {
  return new (Ctx) MCSectionPECOFF(Name, IsDirective, K, Ctx);
}

