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
  AMDGPUELFObjectWriter();
protected:
  unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                        bool IsPCRel) const override {
    return Fixup.getKind();
  }

};


} // End anonymous namespace

AMDGPUELFObjectWriter::AMDGPUELFObjectWriter()
  : MCELFObjectTargetWriter(false, 0, 0, false) { }

MCObjectWriter *llvm::createAMDGPUELFObjectWriter(raw_ostream &OS) {
  MCELFObjectTargetWriter *MOTW = new AMDGPUELFObjectWriter();
  return createELFObjectWriter(MOTW, OS, true);
}
