//=====-- ARMTargetAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMTARGETASMINFO_H
#define ARMTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "ARMSubtarget.h"

namespace llvm {

  // Forward declaration.
  class ARMTargetMachine;

  struct ARMTargetAsmInfo : public TargetAsmInfo {
    ARMTargetAsmInfo(const ARMTargetMachine &TM);

    const ARMSubtarget *Subtarget;

    virtual unsigned getInlineAsmLength(const char *Str) const;
    unsigned countArguments(const char *p) const;
    unsigned countString(const char *p) const;
  };


} // namespace llvm

#endif
