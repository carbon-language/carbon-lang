//=====-- ARMTargetAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMTARGETASMINFO_H
#define ARMTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"

#include "ARMSubtarget.h"

namespace llvm {

  // Forward declaration.
  class ARMTargetMachine;

  struct ARMTargetAsmInfo : public virtual TargetAsmInfo {
    explicit ARMTargetAsmInfo(const ARMTargetMachine &TM);

    const ARMSubtarget *Subtarget;

    virtual unsigned getInlineAsmLength(const char *Str) const;
    unsigned countArguments(const char *p) const;
    unsigned countString(const char *p) const;
  };

  struct ARMDarwinTargetAsmInfo : public virtual ARMTargetAsmInfo,
                                  public virtual DarwinTargetAsmInfo {
    explicit ARMDarwinTargetAsmInfo(const ARMTargetMachine &TM);
  };

  struct ARMELFTargetAsmInfo : public virtual ARMTargetAsmInfo,
                               public virtual ELFTargetAsmInfo {
    explicit ARMELFTargetAsmInfo(const ARMTargetMachine &TM);
  };

} // namespace llvm

#endif
