//=====-- PTXMCAsmInfo.h - PTX asm properties -----------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PTXMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_MCASM_INFO_H
#define PTX_MCASM_INFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  class PTXMCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit PTXMCAsmInfo(const Target &T, const StringRef &TT);
  };
} // namespace llvm

#endif // PTX_MCASM_INFO_H
