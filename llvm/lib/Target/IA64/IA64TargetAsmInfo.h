//=====-- IA64TargetAsmInfo.h - IA64 asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the IA64TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef IA64TARGETASMINFO_H
#define IA64TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class IA64TargetMachine;

  struct IA64TargetAsmInfo : public TargetAsmInfo {
    IA64TargetAsmInfo(const IA64TargetMachine &TM);
  };


} // namespace llvm

#endif
