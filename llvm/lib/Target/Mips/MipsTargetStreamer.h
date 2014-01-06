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

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MipsTargetStreamer : public MCTargetStreamer {
  virtual void anchor();

public:
  virtual void emitMipsHackELFFlags(unsigned Flags) = 0;
  virtual void emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val) = 0;
  virtual void emitDirectiveAbiCalls() = 0;
  virtual void emitDirectiveOptionPic0() = 0;
};

// This part is for ascii assembly output
class MipsTargetAsmStreamer : public MipsTargetStreamer {
  formatted_raw_ostream &OS;

public:
  MipsTargetAsmStreamer(formatted_raw_ostream &OS);
  virtual void emitMipsHackELFFlags(unsigned Flags);
  virtual void emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveOptionPic0();
};

// This part is for ELF object output
class MipsTargetELFStreamer : public MipsTargetStreamer {
public:
  MCELFStreamer &getStreamer();
  MipsTargetELFStreamer();
  // FIXME: emitMipsHackELFFlags() will be removed from this class.
  virtual void emitMipsHackELFFlags(unsigned Flags);
  virtual void emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveOptionPic0();
};
}
#endif
