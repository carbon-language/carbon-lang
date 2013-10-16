//===-- MSP430MCAsmInfo.h - MSP430 asm properties --------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MSP430MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MSP430TARGETASMINFO_H
#define MSP430TARGETASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
  class StringRef;

  class MSP430MCAsmInfo : public MCAsmInfoELF {
    virtual void anchor();
  public:
    explicit MSP430MCAsmInfo(StringRef TT);
  };

} // namespace llvm

#endif
