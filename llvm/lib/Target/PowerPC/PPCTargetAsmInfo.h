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

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class PPCTargetMachine;
  
  struct PPCTargetAsmInfo : public TargetAsmInfo {
    explicit PPCTargetAsmInfo(const PPCTargetMachine &TM);
  };

  struct PPCDarwinTargetAsmInfo : public PPCTargetAsmInfo {
    explicit PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
  };

  struct PPCLinuxTargetAsmInfo : public PPCTargetAsmInfo {
    explicit PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
  };

} // namespace llvm

#endif
