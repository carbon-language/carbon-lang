//=== MipsELFStreamer.h - MipsELFStreamer ------------------------------===//
//
//                    The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENCE.TXT for details.
//
//===-------------------------------------------------------------------===//
#ifndef MIPSELFSTREAMER_H_
#define MIPSELFSTREAMER_H_

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {
class MipsAsmPrinter;
class MipsSubtarget;
class MCSymbol;

class MipsELFStreamer : public MCELFStreamer {
public:
  MipsELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter,
                  bool RelaxAll, bool NoExecStack)
    : MCELFStreamer(SK_MipsELFStreamer, Context, TAB, OS, Emitter) {
  }

  ~MipsELFStreamer() {}
  void emitELFHeaderFlagsCG(const MipsSubtarget &Subtarget);
  void emitMipsSTOCG(const MipsSubtarget &Subtarget,
                     MCSymbol *Sym,
                     unsigned Val);

  static bool classof(const MCStreamer *S) {
    return S->getKind() == SK_MipsELFStreamer;
  }
};

  MCELFStreamer* createMipsELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                       raw_ostream &OS, MCCodeEmitter *Emitter,
                                       bool RelaxAll, bool NoExecStack);
}

#endif /* MIPSELFSTREAMER_H_ */
