//=====-- PIC16TargetAsmInfo.h - PIC16 asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PIC16TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16TARGETASMINFO_H
#define PIC16TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class PIC16TargetMachine;

  struct PIC16TargetAsmInfo : public TargetAsmInfo {
    PIC16TargetAsmInfo(const PIC16TargetMachine &TM);
    public :
  };

} // namespace llvm

#endif
