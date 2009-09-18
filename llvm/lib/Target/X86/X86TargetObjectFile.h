//===-- llvm/Target/X86/X86TargetObjectFile.h - X86 Object Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_X86_TARGETOBJECTFILE_H
#define LLVM_TARGET_X86_TARGETOBJECTFILE_H

#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
  
  /// X8632_MachoTargetObjectFile - This TLOF implementation is used for
  /// Darwin/x86-32.
  class X8632_MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
  public:
    
    virtual const MCExpr *
    getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                     MachineModuleInfo *MMI,
                                     bool &IsIndirect, bool &IsPCRel) const;
  };
  
  /// X8664_MachoTargetObjectFile - This TLOF implementation is used for
  /// Darwin/x86-64.
  class X8664_MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
  public:

    virtual const MCExpr *
    getSymbolForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                     MachineModuleInfo *MMI,
                                     bool &IsIndirect, bool &IsPCRel) const;
  };
} // end namespace llvm

#endif
