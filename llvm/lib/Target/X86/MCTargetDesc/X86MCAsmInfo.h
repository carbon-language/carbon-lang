//===-- X86MCAsmInfo.h - X86 asm properties --------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the X86MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCASMINFO_H
#define LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/MC/MCAsmInfoDarwin.h"
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class TargetTuple;

class X86MCAsmInfoDarwin : public MCAsmInfoDarwin {
  virtual void anchor();

public:
  explicit X86MCAsmInfoDarwin(const TargetTuple &TT);
};

struct X86_64MCAsmInfoDarwin : public X86MCAsmInfoDarwin {
  explicit X86_64MCAsmInfoDarwin(const TargetTuple &TT);
  const MCExpr *
  getExprForPersonalitySymbol(const MCSymbol *Sym, unsigned Encoding,
                              MCStreamer &Streamer) const override;
};

class X86ELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit X86ELFMCAsmInfo(const TargetTuple &TT);
};

class X86MCAsmInfoMicrosoft : public MCAsmInfoMicrosoft {
  void anchor() override;

public:
  explicit X86MCAsmInfoMicrosoft(const TargetTuple &TT);
};

class X86MCAsmInfoGNUCOFF : public MCAsmInfoGNUCOFF {
  void anchor() override;

public:
  explicit X86MCAsmInfoGNUCOFF(const TargetTuple &TT);
};
} // namespace llvm

#endif
