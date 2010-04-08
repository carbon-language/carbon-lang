//====-- SystemZMCAsmInfo.h - SystemZ asm properties -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SystemZMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SystemZTARGETASMINFO_H
#define SystemZTARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  struct SystemZMCAsmInfo : public MCAsmInfo {
    explicit SystemZMCAsmInfo(const Target &T, const StringRef &TT);
    virtual const MCSection *getNonexecutableStackSection(MCContext &Ctx) const;
  };
  
} // namespace llvm

#endif
