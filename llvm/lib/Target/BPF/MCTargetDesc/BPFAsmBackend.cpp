//===-- BPFAsmBackend.cpp - BPF Assembler Backend -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/EndianStream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

namespace {

class BPFAsmBackend : public MCAsmBackend {
public:
  BPFAsmBackend(support::endianness Endian) : MCAsmBackend(Endian) {}
  ~BPFAsmBackend() override = default;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  // No instruction requires relaxation
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    return false;
  }

  unsigned getNumFixupKinds() const override { return 1; }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;
};

} // end anonymous namespace

bool BPFAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                 const MCSubtargetInfo *STI) const {
  if ((Count % 8) != 0)
    return false;

  for (uint64_t i = 0; i < Count; i += 8)
    support::endian::write<uint64_t>(OS, 0x15000000, Endian);

  return true;
}

void BPFAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                               const MCValue &Target,
                               MutableArrayRef<char> Data, uint64_t Value,
                               bool IsResolved,
                               const MCSubtargetInfo *STI) const {
  if (Fixup.getKind() == FK_SecRel_8) {
    // The Value is 0 for global variables, and the in-section offset
    // for static variables. Write to the immediate field of the inst.
    assert(Value <= UINT32_MAX);
    support::endian::write<uint32_t>(&Data[Fixup.getOffset() + 4],
                                     static_cast<uint32_t>(Value),
                                     Endian);
  } else if (Fixup.getKind() == FK_Data_4) {
    support::endian::write<uint32_t>(&Data[Fixup.getOffset()], Value, Endian);
  } else if (Fixup.getKind() == FK_Data_8) {
    support::endian::write<uint64_t>(&Data[Fixup.getOffset()], Value, Endian);
  } else if (Fixup.getKind() == FK_PCRel_4) {
    Value = (uint32_t)((Value - 8) / 8);
    if (Endian == support::little) {
      Data[Fixup.getOffset() + 1] = 0x10;
      support::endian::write32le(&Data[Fixup.getOffset() + 4], Value);
    } else {
      Data[Fixup.getOffset() + 1] = 0x1;
      support::endian::write32be(&Data[Fixup.getOffset() + 4], Value);
    }
  } else {
    assert(Fixup.getKind() == FK_PCRel_2);
    Value = (uint16_t)((Value - 8) / 8);
    support::endian::write<uint16_t>(&Data[Fixup.getOffset() + 2], Value,
                                     Endian);
  }
}

std::unique_ptr<MCObjectTargetWriter>
BPFAsmBackend::createObjectTargetWriter() const {
  return createBPFELFObjectWriter(0);
}

MCAsmBackend *llvm::createBPFAsmBackend(const Target &T,
                                        const MCSubtargetInfo &STI,
                                        const MCRegisterInfo &MRI,
                                        const MCTargetOptions &) {
  return new BPFAsmBackend(support::little);
}

MCAsmBackend *llvm::createBPFbeAsmBackend(const Target &T,
                                          const MCSubtargetInfo &STI,
                                          const MCRegisterInfo &MRI,
                                          const MCTargetOptions &) {
  return new BPFAsmBackend(support::big);
}
