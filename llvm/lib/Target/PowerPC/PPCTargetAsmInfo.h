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

  struct DarwinTargetAsmInfo : public TargetAsmInfo {
    DarwinTargetAsmInfo(const PPCTargetMachine &TM);
    
    /// getSectionForFunction - Return the section that we should emit the
    /// specified function body into.  This defaults to 'TextSection'.  This
    /// should most likely be overridden by the target to put linkonce/weak
    /// functions into special sections.
    virtual const char *getSectionForFunction(const Function &F) const;
  };

} // namespace llvm

#endif
