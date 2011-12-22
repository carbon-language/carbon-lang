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

// Prepare value for the target space for it
static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {

  // Add/subtract and shift
  switch (Kind) {
  default:
    return 0;
  case FK_GPRel_4:
  case FK_Data_4:
  case Mips::fixup_Mips_LO16:
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
  case Mips::fixup_Mips_HI16:
  case Mips::fixup_Mips_GOT_Local:
    // Get the higher 16-bits. Also add 1 if bit 15 is 1.
    Value = (Value >> 16) + ((Value & 0x8000) != 0);
    break;
  }

  return Value;
}

namespace {
class MipsAsmBackend : public MCAsmBackend {
public:
  MipsAsmBackend(const Target &T) : MCAsmBackend() {}

  /// ApplyFixup - Apply the \arg Value for given \arg Fixup into the provided
  /// data fragment, at the offset specified by the fixup and following the
  /// fixup kind as appropriate.
  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
    MCFixupKind Kind = Fixup.getKind();
    Value = adjustFixupValue((unsigned)Kind, Value);

    if (!Value)
      return; // Doesn't change encoding.

    unsigned Offset = Fixup.getOffset();
    // FIXME: The below code will not work across endian models
    // How many bytes/bits are we fixing up?
    unsigned NumBytes = ((getFixupKindInfo(Kind).TargetSize-1)/8)+1;
    uint64_t Mask = ((uint64_t)1 << getFixupKindInfo(Kind).TargetSize) - 1;

    // Grab current value, if any, from bits.
    uint64_t CurVal = 0;
    for (unsigned i = 0; i != NumBytes; ++i)
      CurVal |= ((uint8_t)Data[Offset + i]) << (i * 8);

    CurVal = (CurVal & ~Mask) | ((CurVal + Value) & Mask);

    // Write out the bytes back to the code/data bits.
    // First the unaffected bits and then the fixup.
    for (unsigned i = 0; i != NumBytes; ++i) {
      Data[Offset + i] = uint8_t((CurVal >> (i * 8)) & 0xff);
    }
}

  unsigned getNumFixupKinds() const { return Mips::NumTargetFixupKinds; }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[Mips::NumTargetFixupKinds] = {
      // This table *must* be in same the order of fixup_* kinds in
      // MipsFixupKinds.h.
      //
      // name                    offset  bits  flags
      { "fixup_Mips_16",           0,     16,   0 },
      { "fixup_Mips_32",           0,     32,   0 },
      { "fixup_Mips_REL32",        0,     32,   0 },
      { "fixup_Mips_26",           0,     26,   0 },
      { "fixup_Mips_HI16",         0,     16,   0 },
      { "fixup_Mips_LO16",         0,     16,   0 },
      { "fixup_Mips_GPREL16",      0,     16,   0 },
      { "fixup_Mips_LITERAL",      0,     16,   0 },
      { "fixup_Mips_GOT_Global",   0,     16,   0 },
      { "fixup_Mips_GOT_Local",    0,     16,   0 },
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
      { "fixup_Mips_TLSLDM",       0,     16,   0 },
      { "fixup_Mips_DTPREL_HI",    0,     16,   0 },
      { "fixup_Mips_DTPREL_LO",    0,     16,   0 },
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
    return false;
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
  uint8_t OSABI;

  MipsEB_AsmBackend(const Target &T, uint8_t _OSABI)
    : MipsAsmBackend(T), OSABI(_OSABI) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createMipsELFObjectWriter(OS, /*IsLittleEndian*/ false, OSABI);
  }
};

class MipsEL_AsmBackend : public MipsAsmBackend {
public:
  uint8_t OSABI;

  MipsEL_AsmBackend(const Target &T, uint8_t _OSABI)
    : MipsAsmBackend(T), OSABI(_OSABI) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createMipsELFObjectWriter(OS, /*IsLittleEndian*/ true, OSABI);
  }
};
} // namespace

MCAsmBackend *llvm::createMipsAsmBackend(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  // just return little endian for now
  //
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(Triple(TT).getOS());
  return new MipsEL_AsmBackend(T, OSABI);
}
