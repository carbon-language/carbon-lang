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
class MipsSubtarget;

class MipsELFStreamer : public MCELFStreamer {
public:
  MipsELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter,
                  bool RelaxAll, bool NoExecStack)
    : MCELFStreamer(Context, TAB, OS, Emitter) {
  }

  ~MipsELFStreamer() {}
  void emitELFHeaderFlagsCG(const MipsSubtarget &Subtarget);
//  void emitELFHeaderFlagCG(unsigned Val);
};

  MCELFStreamer* createMipsELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                       raw_ostream &OS, MCCodeEmitter *Emitter,
                                       bool RelaxAll, bool NoExecStack);
}

#endif /* MIPSELFSTREAMER_H_ */
