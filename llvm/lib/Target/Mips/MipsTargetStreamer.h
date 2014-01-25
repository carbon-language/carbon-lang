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
  virtual void emitDirectiveSetMicroMips() = 0;
  virtual void emitDirectiveSetNoMicroMips() = 0;
  virtual void emitDirectiveSetMips16() = 0;
  virtual void emitDirectiveSetNoMips16() = 0;
  virtual void emitDirectiveEnt(const MCSymbol &Symbol) = 0;
  virtual void emitDirectiveAbiCalls() = 0;
  virtual void emitDirectiveOptionPic0() = 0;
};

// This part is for ascii assembly output
class MipsTargetAsmStreamer : public MipsTargetStreamer {
  formatted_raw_ostream &OS;

public:
  MipsTargetAsmStreamer(formatted_raw_ostream &OS);
  virtual void emitMipsHackELFFlags(unsigned Flags);
  virtual void emitDirectiveSetMicroMips();
  virtual void emitDirectiveSetNoMicroMips();
  virtual void emitDirectiveSetMips16();
  virtual void emitDirectiveSetNoMips16();
  virtual void emitDirectiveEnt(const MCSymbol &Symbol);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveOptionPic0();
};

// This part is for ELF object output
class MipsTargetELFStreamer : public MipsTargetStreamer {
  bool MicroMipsEnabled;

public:
  bool isMicroMipsEnabled() const { return MicroMipsEnabled; }
  MCELFStreamer &getStreamer();
  MipsTargetELFStreamer();

  virtual void emitLabel(MCSymbol *Symbol) LLVM_OVERRIDE;

  // FIXME: emitMipsHackELFFlags() will be removed from this class.
  virtual void emitMipsHackELFFlags(unsigned Flags);
  virtual void emitDirectiveSetMicroMips();
  virtual void emitDirectiveSetNoMicroMips();
  virtual void emitDirectiveSetMips16();
  virtual void emitDirectiveSetNoMips16();
  virtual void emitDirectiveEnt(const MCSymbol &Symbol);
  virtual void emitDirectiveAbiCalls();
  virtual void emitDirectiveOptionPic0();
};
}
#endif
