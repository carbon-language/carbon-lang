//===-- MipsASMBackend.cpp -  ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsAsmBackend and MipsELFObjectWriter classes.
//
//===----------------------------------------------------------------------===//
//

#include "MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {

  // Add/subtract and shift
  switch (Kind) {
  default:
    break;
  case Mips::fixup_Mips_PC16:
    // So far we are only using this type for branches.
    // For branches we start 1 instruction after the branch
    // so the displacement will be one instruction size less.
    Value -= 4;
    // The displacement is then divided by 4 to give us an 18 bit
    // address range.
    Value >>= 2;
    break;
  case Mips::fixup_Mips_26:
    // So far we are only using this type for jumps.
    // The displacement is then divided by 4 to give us an 28 bit
    // address range.
    Value >>= 2;
    break;
  }

  // Mask off value for placement as an operand
  switch (Kind) {
  default:
    break;
  case FK_GPRel_4:
  case FK_Data_4:
    Value &= 0xffffffff;
    break;
  case Mips::fixup_Mips_26:
    Value &= 0x03ffffff;
    break;
  case Mips::fixup_Mips_LO16:
  case Mips::fixup_Mips_PC16:
    Value &= 0x0000ffff;
    break;
  case Mips::fixup_Mips_HI16:
    Value >>= 16;
    break;
  }

  return Value;
}

namespace {

class MipsELFObjectWriter : public MCELFObjectTargetWriter {
public:
  MipsELFObjectWriter(bool is64Bit, Triple::OSType OSType, uint16_t EMachine,
                      bool HasRelocationAddend)
    : MCELFObjectTargetWriter(is64Bit, OSType, EMachine,
                              HasRelocationAddend) {}
};

class MipsAsmBackend : public MCAsmBackend {
public:
  MipsAsmBackend(const Target &T) : MCAsmBackend() {}

  /// ApplyFixup - Apply the \arg Value for given \arg Fixup into the provided
  /// data fragment, at the offset specified by the fixup and following the
  /// fixup kind as appropriate.
  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
    unsigned Kind = (unsigned)Fixup.getKind();
    Value = adjustFixupValue(Kind, Value);

    if (!Value)
      return;           // Doesn't change encoding.

    unsigned Offset = Fixup.getOffset();
    switch (Kind) {
    default:
      llvm_unreachable("Unknown fixup kind!");
    case Mips::fixup_Mips_GOT16: // This will be fixed up at link time
     break;
    case FK_GPRel_4:
    case FK_Data_4:
    case Mips::fixup_Mips_26:
    case Mips::fixup_Mips_LO16:
    case Mips::fixup_Mips_PC16:
    case Mips::fixup_Mips_HI16:
      // For each byte of the fragment that the fixup touches, mask i
      // the fixup value. The Value has been "split up" into the appr
      // bitfields above.
      for (unsigned i = 0; i != 4; ++i) // FIXME - Need to support 2 and 8 bytes
        Data[Offset + i] += uint8_t((Value >> (i * 8)) & 0xff);
      break;
    }
  }

  unsigned getNumFixupKinds() const { return Mips::NumTargetFixupKinds; }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[Mips::NumTargetFixupKinds] = {
      // This table *must* be in the order that the fixup_* kinds a
      // MipsFixupKinds.h.
      //
      // name                    offset  bits  flags
      { "fixup_Mips_NONE",         0,      0,   0 },
      { "fixup_Mips_16",           0,     16,   0 },
      { "fixup_Mips_32",           0,     32,   0 },
      { "fixup_Mips_REL32",        0,     32,   0 },
      { "fixup_Mips_26",           0,     26,   0 },
      { "fixup_Mips_HI16",         0,     16,   0 },
      { "fixup_Mips_LO16",         0,     16,   0 },
      { "fixup_Mips_GPREL16",      0,     16,   0 },
      { "fixup_Mips_LITERAL",      0,     16,   0 },
      { "fixup_Mips_GOT16",        0,     16,   0 },
      { "fixup_Mips_PC16",         0,     16,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_Mips_CALL16",       0,     16,   0 },
      { "fixup_Mips_GPREL32",      0,     32,   0 },
      { "fixup_Mips_SHIFT5",       6,      5,   0 },
      { "fixup_Mips_SHIFT6",       6,      5,   0 },
      { "fixup_Mips_64",           0,     64,   0 },
      { "fixup_Mips_TLSGD",        0,     16,   0 },
      { "fixup_Mips_GOTTPREL",     0,     16,   0 },
      { "fixup_Mips_TPREL_HI",     0,     16,   0 },
      { "fixup_Mips_TPREL_LO",     0,     16,   0 },
      { "fixup_Mips_Branch_PCRel", 0,     16,  MCFixupKindInfo::FKF_IsPCRel }
    };

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  /// @name Target Relaxation Interfaces
  /// @{

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \param Inst - The instruction to test.
  bool MayNeedRelaxation(const MCInst &Inst) const {
    return false;
  }

  /// fixupNeedsRelaxation - Target specific predicate for whether a given
  /// fixup requires the associated instruction to be relaxed.
  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value,
                            const MCInstFragment *DF,
                            const MCAsmLayout &Layout) const {
    // FIXME.
    assert(0 && "RelaxInstruction() unimplemented");
  }

  /// RelaxInstruction - Relax the instruction in the given fragment
  /// to the next wider instruction.
  ///
  /// \param Inst - The instruction to relax, which may be the same
  /// as the output.
  /// \parm Res [output] - On return, the relaxed instruction.
  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
  }
  
  /// @}

  /// WriteNopData - Write an (optimal) nop sequence of Count bytes
  /// to the given output. If the target cannot generate such a sequence,
  /// it should return an error.
  ///
  /// \return - True on success.
  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
    return true;
  }
};

class MipsEB_AsmBackend : public MipsAsmBackend {
public:
  Triple::OSType OSType;

  MipsEB_AsmBackend(const Target &T, Triple::OSType _OSType)
    : MipsAsmBackend(T), OSType(_OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(createELFObjectTargetWriter(),
                                 OS, /*IsLittleEndian*/ false);
  }

  MCELFObjectTargetWriter *createELFObjectTargetWriter() const {
    return new MipsELFObjectWriter(false, OSType, ELF::EM_MIPS, false);
  }
};

class MipsEL_AsmBackend : public MipsAsmBackend {
public:
  Triple::OSType OSType;

  MipsEL_AsmBackend(const Target &T, Triple::OSType _OSType)
    : MipsAsmBackend(T), OSType(_OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(createELFObjectTargetWriter(),
                                 OS, /*IsLittleEndian*/ true);
  }

  MCELFObjectTargetWriter *createELFObjectTargetWriter() const {
    return new MipsELFObjectWriter(false, OSType, ELF::EM_MIPS, false);
  }
};
} // namespace

MCAsmBackend *llvm::createMipsAsmBackend(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  // just return little endian for now
  //
  return new MipsEL_AsmBackend(T, Triple(TT).getOS());
}
