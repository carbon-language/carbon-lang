//=====-- SparcTargetAsmInfo.h - Sparc asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SparcTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETASMINFO_H
#define SPARCTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class SparcTargetMachine;

  struct SparcTargetAsmInfo : public TargetAsmInfo {
    SparcTargetAsmInfo(const SparcTargetMachine &TM);
  };


} // namespace llvm

#endif
