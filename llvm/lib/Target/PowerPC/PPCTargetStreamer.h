//===-- PPCTargetStreamer.h - PPC Target Streamer --s-----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETSTREAMER_H
#define PPCTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class PPCTargetStreamer : public MCTargetStreamer {
public:
  PPCTargetStreamer(MCStreamer &S);
  virtual ~PPCTargetStreamer();
  virtual void emitTCEntry(const MCSymbol &S) = 0;
  virtual void emitMachine(StringRef CPU) = 0;
  virtual void emitAbiVersion(int AbiVersion) = 0;
  virtual void emitLocalEntry(MCSymbol *S, const MCExpr *LocalOffset) = 0;
};
}

#endif
