//===-- AMDGPUELFObjectWriter.cpp - AMDGPU ELF Writer ----------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"

using namespace llvm;

namespace {

class AMDGPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  AMDGPUELFObjectWriter(const Triple &TT);
protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override {
    return Fixup.getKind();
  }
};


} // End anonymous namespace

AMDGPUELFObjectWriter::AMDGPUELFObjectWriter(const Triple &TT)
  : MCELFObjectTargetWriter(TT.getArch() == Triple::amdgcn, // Is64Bit
                            ELF::ELFOSABI_AMDGPU_HSA,
                            ELF::EM_AMDGPU,
                            // HasRelocationAddend
                            TT.getOS() == Triple::AMDHSA) {}


MCObjectWriter *llvm::createAMDGPUELFObjectWriter(const Triple &TT,
                                                  raw_pwrite_stream &OS) {
  MCELFObjectTargetWriter *MOTW = new AMDGPUELFObjectWriter(TT);
  return createELFObjectWriter(MOTW, OS, true);
}
