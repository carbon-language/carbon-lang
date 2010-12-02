//===-- ARMAsmBackend.cpp - ARM Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMFixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

namespace {
class ARMAsmBackend : public TargetAsmBackend {
public:
  ARMAsmBackend(const Target &T) : TargetAsmBackend() {}

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
  // FIXME: Zero fill for now. That's not right, but at least will get the
  // section size right.
  for (uint64_t i = 0; i != Count; ++i)
    OW->Write8(0);
  return true;
}

static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_4:
  case ARM::fixup_arm_movt_hi16:
  case ARM::fixup_arm_movw_lo16:
    return Value;
  case ARM::fixup_arm_ldst_pcrel_12: {
    bool isAdd = true;
    // ARM PC-relative values are offset by 8.
    Value -= 8;
    if ((int64_t)Value < 0) {
      Value = -Value;
      isAdd = false;
    }
    assert ((Value < 4096) && "Out of range pc-relative fixup value!");
    Value |= isAdd << 23;
    return Value;
  }
  case ARM::fixup_arm_adr_pcrel_12: {
    // ARM PC-relative values are offset by 8.
    Value -= 8;
    unsigned opc = 4; // bits {24-21}. Default to add: 0b0100
    if ((int64_t)Value < 0) {
      Value = -Value;
      opc = 2; // 0b0010
    }
    assert(ARM_AM::getSOImmVal(Value) != -1 &&
           "Out of range pc-relative fixup value!");
    // Encode the immediate and shift the opcode into place.
    return ARM_AM::getSOImmVal(Value) | (opc << 21);
  }
  case ARM::fixup_arm_branch:
    // These values don't encode the low two bits since they're always zero.
    // Offset by 8 just as above.
    return (Value - 8) >> 2;
  case ARM::fixup_arm_pcrel_10: {
    // Offset by 8 just as above.
    Value = Value - 8;
    bool isAdd = true;
    if ((int64_t)Value < 0) {
      Value = -Value;
      isAdd = false;
    }
    // These values don't encode the low two bits since they're always zero.
    Value >>= 2;
    assert ((Value < 256) && "Out of range pc-relative fixup value!");
    Value |= isAdd << 23;
    return Value;
  }
  }
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
  uint32_t Mask = 0;
  // Fixme: 2 for Thumb
  unsigned NumBytes = 4;
  Value = adjustFixupValue(Fixup.getKind(), Value);

  switch (Fixup.getKind()) {
  default: assert(0 && "Unsupported Fixup kind"); break;
  case ARM::fixup_arm_branch: {
    unsigned Lo24 = Value & 0xFFFFFF;
    Mask = ~(0xFFFFFF);
    Value = Lo24;
  }; break;
  case ARM::fixup_arm_movt_hi16:
  case ARM::fixup_arm_movw_lo16: {
    unsigned Hi4 = (Value & 0xF000) >> 12;
    unsigned Lo12 = Value & 0x0FFF;
    // inst{19-16} = Hi4;
    // inst{11-0} = Lo12;
    Value = (Hi4 << 16) | (Lo12);
    Mask = ~(0xF0FFF);
  }; break;
  }

  assert((Fixup.getOffset() % NumBytes == 0)
         && "Offset mod NumBytes is nonzero!");
  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  // The Value has been "split up" into the appropriate bitfields above.
  // Fixme: how to share code with the .td generated code?
  for (unsigned i = 0; i != NumBytes; ++i) {
    DF.getContents()[Fixup.getOffset() + i] &= uint8_t(Mask >> (i * 8));
    DF.getContents()[Fixup.getOffset() + i] |= uint8_t(Value >> (i * 8));
  }
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

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    // FIXME: Subtarget info should be derived. Force v7 for now.
    return createMachObjectWriter(OS, /*Is64Bit=*/false,
                                  object::mach::CTM_ARM,
                                  object::mach::CSARM_V7,
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
  case ARM::fixup_arm_ldst_pcrel_12: return 3;
  case ARM::fixup_arm_pcrel_10: return 3;
  case ARM::fixup_arm_adr_pcrel_12: return 3;
  case ARM::fixup_arm_branch: return 3;
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
