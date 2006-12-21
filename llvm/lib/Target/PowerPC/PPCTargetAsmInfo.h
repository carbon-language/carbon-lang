//=====-- PPCTargetAsmInfo.h - PPC asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the DarwinTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETASMINFO_H
#define PPCTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class PPCTargetMachine;
  
  struct PPCTargetAsmInfo : public TargetAsmInfo {
    PPCTargetAsmInfo(const PPCTargetMachine &TM);
  };

  struct DarwinTargetAsmInfo : public PPCTargetAsmInfo {
    DarwinTargetAsmInfo(const PPCTargetMachine &TM);
  };

  struct LinuxTargetAsmInfo : public PPCTargetAsmInfo {
    LinuxTargetAsmInfo(const PPCTargetMachine &TM);
  };

} // namespace llvm

#endif
