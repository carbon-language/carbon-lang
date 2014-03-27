//===-------- MipsELFStreamer.h - ELF Object Output -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCELFStreamer which allows us to insert some hooks before
// emitting data into an actual object file.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSELFSTREAMER_H
#define MIPSELFSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCSubtargetInfo;

class MipsELFStreamer : public MCELFStreamer {

public:
  MipsELFStreamer(MCContext &Context, MCAsmBackend &MAB, raw_ostream &OS,
                  MCCodeEmitter *Emitter, const MCSubtargetInfo &STI)
      : MCELFStreamer(Context, MAB, OS, Emitter) {}

  virtual ~MipsELFStreamer() {}
};

MCELFStreamer *createMipsELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_ostream &OS, MCCodeEmitter *Emitter,
                                     const MCSubtargetInfo &STI, bool RelaxAll,
                                     bool NoExecStack);
} // namespace llvm.
#endif
