//===-- ARM64TargetObjectFile.h - ARM64 Object Info -*- C++ -------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM64_TARGETOBJECTFILE_H
#define LLVM_TARGET_ARM64_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
class ARM64TargetMachine;

/// This implementation is used for AArch64 ELF targets (Linux in particular).
class ARM64_ELFTargetObjectFile : public TargetLoweringObjectFileELF {
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
};

/// ARM64_MachoTargetObjectFile - This TLOF implementation is used for Darwin.
class ARM64_MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
public:
  const MCExpr *getTTypeGlobalReference(const GlobalValue *GV,
                                        unsigned Encoding, Mangler &Mang,
                                        const TargetMachine &TM,
                                        MachineModuleInfo *MMI,
                                        MCStreamer &Streamer) const override;

  MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                    const TargetMachine &TM,
                                    MachineModuleInfo *MMI) const override;
};

} // end namespace llvm

#endif
