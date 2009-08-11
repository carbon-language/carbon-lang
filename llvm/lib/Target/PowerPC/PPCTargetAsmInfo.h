//=====-- PPCTargetAsmInfo.h - PPC asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the DarwinTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETASMINFO_H
#define PPCTARGETASMINFO_H

#include "PPCTargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

  struct PPCDarwinTargetAsmInfo : public DarwinTargetAsmInfo {
    explicit PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM);
  };

  struct PPCLinuxTargetAsmInfo : public TargetAsmInfo {
    explicit PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM);
  };

} // namespace llvm

#endif
