//===-- PPCTargetObjectFile.h - PPC Object Info -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PPC_TARGETOBJECTFILE_H
#define LLVM_TARGET_PPC_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

  /// PPC64LinuxTargetObjectFile - This implementation is used for
  /// 64-bit PowerPC Linux.
  class PPC64LinuxTargetObjectFile : public TargetLoweringObjectFileELF {

    void Initialize(MCContext &Ctx, const TargetMachine &TM) LLVM_OVERRIDE;

    const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                            SectionKind Kind, Mangler &Mang,
                                            const TargetMachine &TM) const
        LLVM_OVERRIDE;

    /// \brief Describe a TLS variable address within debug info.
    const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const
        LLVM_OVERRIDE;
  };

}  // end namespace llvm

#endif
