//===-- MipsTargetStreamer.h - Mips Target Streamer ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETSTREAMER_H
#define MIPSTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MipsTargetStreamer : public MCTargetStreamer {
public:
  virtual void emitMipsHackELFFlags(unsigned Flags) = 0;
  virtual void emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val) = 0;
};
}

#endif
