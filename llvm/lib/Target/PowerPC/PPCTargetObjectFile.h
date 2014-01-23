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

    virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

    virtual const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                                    SectionKind Kind,
                                                    Mangler *Mang,
                                                    TargetMachine &TM) const;

    /// \brief Describe a TLS variable address within debug info.
    virtual const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const;
  };

}  // end namespace llvm

#endif
