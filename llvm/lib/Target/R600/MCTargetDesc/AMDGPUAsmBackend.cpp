//===-- AMDGPUAsmBackend.cpp - AMDGPU Assembler Backend -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {

class AMDGPUMCObjectWriter : public MCObjectWriter {
public:
  AMDGPUMCObjectWriter(raw_ostream &OS) : MCObjectWriter(OS, true) { }
  virtual void ExecutePostLayoutBinding(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
    //XXX: Implement if necessary.
  }
  virtual void RecordRelocation(const MCAssembler &Asm,
                                const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCFixup &Fixup,
                                MCValue Target, uint64_t &FixedValue) {
    assert(!"Not implemented");
  }

  virtual void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);

};

class AMDGPUAsmBackend : public MCAsmBackend {
public:
  AMDGPUAsmBackend(const Target &T)
    : MCAsmBackend() {}

  virtual unsigned getNumFixupKinds() const { return 0; };
  virtual void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                          uint64_t Value) const;
  virtual bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout) const {
    return false;
  }
  virtual void relaxInstruction(const MCInst &Inst, MCInst &Res) const {
    assert(!"Not implemented");
  }
  virtual bool mayNeedRelaxation(const MCInst &Inst) const { return false; }
  virtual bool writeNopData(uint64_t Count, MCObjectWriter *OW) const {
    return true;
  }
};

} //End anonymous namespace

void AMDGPUMCObjectWriter::WriteObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  for (MCAssembler::iterator I = Asm.begin(), E = Asm.end(); I != E; ++I) {
    Asm.writeSectionData(I, Layout);
  }
}

void AMDGPUAsmBackend::applyFixup(const MCFixup &Fixup, char *Data,
                                  unsigned DataSize, uint64_t Value) const {

  uint16_t *Dst = (uint16_t*)(Data + Fixup.getOffset());
  assert(Fixup.getKind() == FK_PCRel_4);
  *Dst = (Value - 4) / 4;
}

//===----------------------------------------------------------------------===//
// ELFAMDGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFAMDGPUAsmBackend : public AMDGPUAsmBackend {
public:
  ELFAMDGPUAsmBackend(const Target &T) : AMDGPUAsmBackend(T) { }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createAMDGPUELFObjectWriter(OS);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createAMDGPUAsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           StringRef TT,
                                           StringRef CPU) {
  return new ELFAMDGPUAsmBackend(T);
}
