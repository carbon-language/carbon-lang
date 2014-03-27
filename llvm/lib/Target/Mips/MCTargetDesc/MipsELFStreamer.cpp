//===-------- MipsELFStreamer.cpp - ELF Object Output ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFStreamer.h"

namespace llvm {
MCELFStreamer *createMipsELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_ostream &OS, MCCodeEmitter *Emitter,
                                     const MCSubtargetInfo &STI, bool RelaxAll,
                                     bool NoExecStack) {
  return new MipsELFStreamer(Context, MAB, OS, Emitter, STI);
}
}
