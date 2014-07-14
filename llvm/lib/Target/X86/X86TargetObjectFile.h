//===-- X86TargetObjectFile.h - X86 Object Info -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_X86_TARGETOBJECTFILE_H
#define LLVM_TARGET_X86_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {

  /// X86_64MachoTargetObjectFile - This TLOF implementation is used for Darwin
  /// x86-64.
  class X86_64MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
  public:
    const MCExpr *
    getTTypeGlobalReference(const GlobalValue *GV, unsigned Encoding,
                            Mangler &Mang, const TargetMachine &TM,
                            MachineModuleInfo *MMI,
                            MCStreamer &Streamer) const override;

    // getCFIPersonalitySymbol - The symbol that gets passed to
    // .cfi_personality.
    MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                      const TargetMachine &TM,
                                      MachineModuleInfo *MMI) const override;
  };

  /// X86LinuxTargetObjectFile - This implementation is used for linux x86
  /// and x86-64.
  class X86LinuxTargetObjectFile : public TargetLoweringObjectFileELF {
    void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

    /// \brief Describe a TLS variable address within debug info.
    const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const override;
  };

  /// \brief This implementation is used for Windows targets on x86 and x86-64.
  class X86WindowsTargetObjectFile : public TargetLoweringObjectFileCOFF {
    const MCExpr *
    getExecutableRelativeSymbol(const ConstantExpr *CE, Mangler &Mang,
                                const TargetMachine &TM) const override;

    /// \brief Given a mergeable constant with the specified size and relocation
    /// information, return a section that it should be placed in.
    const MCSection *getSectionForConstant(SectionKind Kind,
                                           const Constant *C) const override;
  };

} // end namespace llvm

#endif
