//=====-- ARM64MCAsmInfo.h - ARM64 asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARM64MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64TARGETASMINFO_H
#define ARM64TARGETASMINFO_H

#include "llvm/MC/MCAsmInfoDarwin.h"

namespace llvm {
class Target;
class StringRef;
class MCStreamer;
struct ARM64MCAsmInfoDarwin : public MCAsmInfoDarwin {
  explicit ARM64MCAsmInfoDarwin();
  virtual const MCExpr *getExprForPersonalitySymbol(const MCSymbol *Sym,
                                                    unsigned Encoding,
                                                    MCStreamer &Streamer) const;
};

struct ARM64MCAsmInfoELF : public MCAsmInfo {
  explicit ARM64MCAsmInfoELF(StringRef TT);
};

} // namespace llvm

#endif
