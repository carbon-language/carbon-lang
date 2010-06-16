//===- lib/MC/MCObjectStreamer.cpp - Object File MCStreamer Interface -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectStreamer.h"

#include "llvm/MC/MCAssembler.h"
using namespace llvm;

MCObjectStreamer::MCObjectStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                   raw_ostream &_OS, MCCodeEmitter *_Emitter)
  : MCStreamer(Context), Assembler(new MCAssembler(Context, TAB,
                                                   *_Emitter, _OS)),
    CurSectionData(0)
{
}

MCObjectStreamer::~MCObjectStreamer() {
  delete Assembler;
}

void MCObjectStreamer::SwitchSection(const MCSection *Section) {
  assert(Section && "Cannot switch to a null section!");

  // If already in this section, then this is a noop.
  if (Section == CurSection) return;

  CurSection = Section;
  CurSectionData = &getAssembler().getOrCreateSectionData(*Section);
}

void MCObjectStreamer::Finish() {
  getAssembler().Finish();
}
