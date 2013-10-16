//====-- SystemZMCAsmInfo.h - SystemZ asm properties -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SystemZTARGETASMINFO_H
#define SystemZTARGETASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class StringRef;

class SystemZMCAsmInfo : public MCAsmInfoELF {
public:
  explicit SystemZMCAsmInfo(StringRef TT);

  // Override MCAsmInfo;
  virtual const MCSection *getNonexecutableStackSection(MCContext &Ctx) const
    LLVM_OVERRIDE;
};

} // namespace llvm

#endif
