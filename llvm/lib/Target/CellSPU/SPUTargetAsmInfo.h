//=====-- SPUTargetAsmInfo.h - Cell SPU asm properties --------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SPUTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETASMINFO_H
#define PPCTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class SPUTargetMachine;

  struct SPUTargetAsmInfo : public TargetAsmInfo {
    SPUTargetAsmInfo(const SPUTargetMachine &TM);
  };

} // namespace llvm

#endif
