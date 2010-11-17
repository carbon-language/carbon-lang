//===-- ARMAsmBackend.cpp - ARM Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmBackend.h"
#include "ARM.h"
#include "ARMFixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

namespace {
class ARMAsmBackend : public TargetAsmBackend {
public:
  ARMAsmBackend(const Target &T) : TargetAsmBackend(T) {}

  bool MayNeedRelaxation(const MCInst &Inst) const;

  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const;

  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const;

  unsigned getPointerSize() const {
    return 4;
  }
};
} // end anonymous namespace

bool ARMAsmBackend::MayNeedRelaxation(const MCInst &Inst) const {
  // FIXME: Thumb targets, different move constant targets..
  return false;
}

void ARMAsmBackend::RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
  assert(0 && "ARMAsmBackend::RelaxInstruction() unimplemented");
  return;
}

bool ARMAsmBackend::WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
//  if ((Count % 4) != 0) {
//    // Fixme: % 2 for Thumb?
//    return false;
//  }
  // FIXME: Zero fill for now. That's not right, but at least will get the
  // section size right.
  for (uint64_t i = 0; i != Count; ++i)
    OW->Write8(0);
  return true;
}

namespace {
// FIXME: This should be in a separate file.
// ELF is an ELF of course...
class ELFARMAsmBackend : public ARMAsmBackend {
  MCELFObjectFormat Format;

public:
  Triple::OSType OSType;
  ELFARMAsmBackend(const Target &T, Triple::OSType _OSType)
    : ARMAsmBackend(T), OSType(_OSType) {
    HasScatteredSymbols = true;
  }

  virtual const MCObjectFormat &getObjectFormat() const {
    return Format;
  }

  void ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                  uint64_t Value) const;

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionELF &SE = static_cast<const MCSectionELF&>(Section);
    return SE.getType() == MCSectionELF::SHT_NOBITS;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(OS, /*Is64Bit=*/false,
                                 OSType, ELF::EM_ARM,
                                 /*IsLittleEndian=*/true,
                                 /*HasRelocationAddend=*/false);
  }
};

// Fixme: can we raise this to share code between Darwin and ELF?
void ELFARMAsmBackend::ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                                  uint64_t Value) const {
  assert(0 && "ELFARMAsmBackend::ApplyFixup() unimplemented");
}

namespace {
// FIXME: This should be in a separate file.
class DarwinARMAsmBackend : public ARMAsmBackend {
  MCMachOObjectFormat Format;
public:
  DarwinARMAsmBackend(const Target &T) : ARMAsmBackend(T) {
    HasScatteredSymbols = true;
  }

  virtual const MCObjectFormat &getObjectFormat() const {
    return Format;
  }

  void ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                  uint64_t Value) const;

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    return (SMO.getType() == MCSectionMachO::S_ZEROFILL ||
            SMO.getType() == MCSectionMachO::S_GB_ZEROFILL ||
            SMO.getType() == MCSectionMachO::S_THREAD_LOCAL_ZEROFILL);
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    // FIXME: Subtarget info should be derived. Force v7 for now.
    return createMachObjectWriter(OS, /*Is64Bit=*/false, MachO::CPUTypeARM,
                                  MachO::CPUSubType_ARM_V7,
                                  /*IsLittleEndian=*/true);
  }

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    return false;
  }
};
} // end anonymous namespace

static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  default: llvm_unreachable("Unknown fixup kind!");
  case FK_Data_4: return 4;
  case ARM::fixup_arm_pcrel_12: return 2;
  case ARM::fixup_arm_vfp_pcrel_12: return 1;
  case ARM::fixup_arm_branch: return 3;
  }
}

static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_4:
    return Value;
  case ARM::fixup_arm_pcrel_12:
    // ARM PC-relative values are offset by 8.
    return Value - 8;
  case ARM::fixup_arm_branch:
  case ARM::fixup_arm_vfp_pcrel_12:
    // These values don't encode the low two bits since they're always zero.
    // Offset by 8 just as above.
    return (Value - 8) >> 2;
  }
}

void DarwinARMAsmBackend::ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                                     uint64_t Value) const {
  unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
  Value = adjustFixupValue(Fixup.getKind(), Value);

  assert(Fixup.getOffset() + NumBytes <= DF.getContents().size() &&
         "Invalid fixup offset!");
  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  for (unsigned i = 0; i != NumBytes; ++i)
    DF.getContents()[Fixup.getOffset() + i] |= uint8_t(Value >> (i * 8));
}
} // end anonymous namespace

TargetAsmBackend *llvm::createARMAsmBackend(const Target &T,
                                            const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinARMAsmBackend(T);
  case Triple::MinGW32:
  case Triple::Cygwin:
  case Triple::Win32:
    assert(0 && "Windows not supported on ARM");
  default:
    return new ELFARMAsmBackend(T, Triple(TT).getOS());
  }
}
