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

MCSection::MCSection(const StringRef &N, MCContext &Ctx) : Name(N) {
  MCSection *&Entry = Ctx.Sections[Name];
  assert(Entry == 0 && "Multiple sections with the same name created");
  Entry = this;
}

MCSection *MCSection::Create(const StringRef &Name, MCContext &Ctx) {
  return new (Ctx) MCSection(Name, Ctx);
}


MCSectionWithKind *
MCSectionWithKind::Create(const StringRef &Name, SectionKind K, MCContext &Ctx){
  return new (Ctx) MCSectionWithKind(Name, K, Ctx);
}
