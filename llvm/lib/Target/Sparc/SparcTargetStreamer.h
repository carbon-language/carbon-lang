//===-- SparcTargetStreamer.h - Sparc Target Streamer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETSTREAMER_H
#define SPARCTARGETSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class SparcTargetStreamer : public MCTargetStreamer {
  virtual void anchor();

public:
  SparcTargetStreamer(MCStreamer &S);
  /// Emit ".register <reg>, #ignore".
  virtual void emitSparcRegisterIgnore(unsigned reg) = 0;
  /// Emit ".register <reg>, #scratch".
  virtual void emitSparcRegisterScratch(unsigned reg) = 0;
};

// This part is for ascii assembly output
class SparcTargetAsmStreamer : public SparcTargetStreamer {
  formatted_raw_ostream &OS;

public:
  SparcTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  virtual void emitSparcRegisterIgnore(unsigned reg);
  virtual void emitSparcRegisterScratch(unsigned reg);

};

// This part is for ELF object output
class SparcTargetELFStreamer : public SparcTargetStreamer {
public:
  SparcTargetELFStreamer(MCStreamer &S);
  MCELFStreamer &getStreamer();
  virtual void emitSparcRegisterIgnore(unsigned reg) {}
  virtual void emitSparcRegisterScratch(unsigned reg) {}
};
} // end namespace llvm

#endif
