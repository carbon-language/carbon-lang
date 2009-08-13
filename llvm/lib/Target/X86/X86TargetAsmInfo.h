//=====-- X86TargetAsmInfo.h - X86 asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the X86TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETASMINFO_H
#define X86TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/COFFTargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"

namespace llvm {
  struct X86DarwinTargetAsmInfo : public DarwinTargetAsmInfo {
    explicit X86DarwinTargetAsmInfo(const Triple &Triple);
  };

  struct X86ELFTargetAsmInfo : public TargetAsmInfo {
    explicit X86ELFTargetAsmInfo(const Triple &Triple);
  };

  struct X86COFFTargetAsmInfo : public COFFTargetAsmInfo {
    explicit X86COFFTargetAsmInfo(const Triple &Triple);
  };

  struct X86WinTargetAsmInfo : public TargetAsmInfo {
    explicit X86WinTargetAsmInfo(const Triple &Triple);
  };

} // namespace llvm

#endif
