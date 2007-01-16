//=====-- X86TargetAsmInfo.h - X86 asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the X86TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETASMINFO_H
#define X86TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class X86TargetMachine;

  struct X86TargetAsmInfo : public TargetAsmInfo {
    X86TargetAsmInfo(const X86TargetMachine &TM);
    
    virtual bool ExpandInlineAsm(CallInst *CI) const;
  private:
    bool LowerToBSwap(CallInst *CI) const;
  };
} // namespace llvm

#endif
