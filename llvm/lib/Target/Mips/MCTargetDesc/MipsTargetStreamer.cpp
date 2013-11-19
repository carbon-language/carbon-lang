//===-- MipsTargetStreamer.cpp - Mips Target Streamer Methods -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetStreamer.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

// pin vtable to this file
void MipsTargetStreamer::anchor() {}

MipsTargetAsmStreamer::MipsTargetAsmStreamer() {}

void MipsTargetAsmStreamer::emitMipsHackELFFlags(unsigned Flags) {
  return;

}
void MipsTargetAsmStreamer::emitSymSTO(MCSymbol *Sym, unsigned Val) {
  return;

}

MipsTargetELFStreamer::MipsTargetELFStreamer() {}

MCELFStreamer &MipsTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(*Streamer);
}

void MipsTargetELFStreamer::emitMipsHackELFFlags(unsigned Flags) {
  MCAssembler &MCA = getStreamer().getAssembler();
  MCA.setELFHeaderEFlags(Flags);
}

// Set a symbol's STO flags
void MipsTargetELFStreamer::emitSymSTO(MCSymbol *Sym, unsigned Val) {
  MCSymbolData &Data = getStreamer().getOrCreateSymbolData(Sym);
  // The "other" values are stored in the last 6 bits of the second byte
  // The traditional defines for STO values assume the full byte and thus
  // the shift to pack it.
  MCELF::setOther(Data, Val >> 2);
}
