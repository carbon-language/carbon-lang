//=====-- PIC16MCAsmInfo.h - PIC16 asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PIC16MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16TARGETASMINFO_H
#define PIC16TARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  class PIC16MCAsmInfo : public MCAsmInfo {
    const char *RomData8bitsDirective;
    const char *RomData16bitsDirective;
    const char *RomData32bitsDirective;
  public:    
    PIC16MCAsmInfo(const Target &T, StringRef TT);
    
    virtual const char *getDataASDirective(unsigned size, unsigned AS) const;
  };

} // namespace llvm

#endif
