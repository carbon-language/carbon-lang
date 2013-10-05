//===-- MipsELFStreamer.cpp - MipsELFStreamer ---------------------------===//
//
//                       The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//
#include "MipsSubtarget.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

namespace {
class MipsELFStreamer : public MCELFStreamer {
public:
  MipsELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &OS,
                  MCCodeEmitter *Emitter, bool RelaxAll, bool NoExecStack)
      : MCELFStreamer(Context, TAB, OS, Emitter) {}

  ~MipsELFStreamer() {}
  void emitMipsHackELFFlags(unsigned Flags);
  void emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val);
};
}

namespace llvm {
MCELFStreamer *createMipsELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                     raw_ostream &OS, MCCodeEmitter *Emitter,
                                     bool RelaxAll, bool NoExecStack) {
  MipsELFStreamer *S =
      new MipsELFStreamer(Context, TAB, OS, Emitter, RelaxAll, NoExecStack);
  return S;
}
} // namespace llvm

void MipsELFStreamer::emitMipsHackELFFlags(unsigned Flags) {
  MCAssembler &MCA = getAssembler();

  MCA.setELFHeaderEFlags(Flags);
}

// Set a symbol's STO flags
void MipsELFStreamer::emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val) {
  MCSymbolData &Data = getOrCreateSymbolData(Sym);
  // The "other" values are stored in the last 6 bits of the second byte
  // The traditional defines for STO values assume the full byte and thus
  // the shift to pack it.
  MCELF::setOther(Data, Val >> 2);
}
