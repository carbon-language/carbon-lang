//===-- SparcAsmBackend.cpp - Sparc Assembler Backend ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmBackend.h"
#include "MCTargetDesc/SparcMCTargetDesc.h"
#include "MCTargetDesc/SparcFixupKinds.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
    return Value;
  case Sparc::fixup_sparc_call30:
    return Value & 0x3fffffff;
  case Sparc::fixup_sparc_br22:
    return Value & 0x3fffff;
  case Sparc::fixup_sparc_br19:
    return Value & 0x1ffff;
  case Sparc::fixup_sparc_hi22:
    return (Value >> 10) & 0x3fffff;
  case Sparc::fixup_sparc_lo10:
    return Value & 0x3ff;
  case Sparc::fixup_sparc_h44:
    return (Value >> 22) & 0x3fffff;
  case Sparc::fixup_sparc_m44:
    return (Value >> 12) & 0x3ff;
  case Sparc::fixup_sparc_l44:
    return Value & 0xfff;
  case Sparc::fixup_sparc_hh:
    return (Value >> 42) & 0x3fffff;
  case Sparc::fixup_sparc_hm:
    return (Value >>32) & 0x3ff;
  }
}

namespace {
  class SparcAsmBackend : public MCAsmBackend {
    const Target &TheTarget;
  public:
    SparcAsmBackend(const Target &T) : MCAsmBackend(), TheTarget(T) {}

    unsigned getNumFixupKinds() const {
      return Sparc::NumTargetFixupKinds;
    }

    const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
      const static MCFixupKindInfo Infos[Sparc::NumTargetFixupKinds] = {
        // name                    offset bits  flags
        { "fixup_sparc_call30",     0,     30,  MCFixupKindInfo::FKF_IsPCRel },
        { "fixup_sparc_br22",       0,     22,  MCFixupKindInfo::FKF_IsPCRel },
        { "fixup_sparc_br19",       0,     19,  MCFixupKindInfo::FKF_IsPCRel },
        { "fixup_sparc_hi22",       0,     22,  0 },
        { "fixup_sparc_lo10",       0,     10,  0 },
        { "fixup_sparc_h44",        0,     22,  0 },
        { "fixup_sparc_m44",        0,     10,  0 },
        { "fixup_sparc_l44",        0,     12,  0 },
        { "fixup_sparc_hh",         0,     21,  0 },
        { "fixup_sparc_hm",         0,     10,  0 },
      };

      if (Kind < FirstTargetFixupKind)
        return MCAsmBackend::getFixupKindInfo(Kind);

      assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
             "Invalid kind!");
      return Infos[Kind - FirstTargetFixupKind];
    }

    bool mayNeedRelaxation(const MCInst &Inst) const {
      // FIXME.
      return false;
    }

    /// fixupNeedsRelaxation - Target specific predicate for whether a given
    /// fixup requires the associated instruction to be relaxed.
    bool fixupNeedsRelaxation(const MCFixup &Fixup,
                              uint64_t Value,
                              const MCRelaxableFragment *DF,
                              const MCAsmLayout &Layout) const {
      // FIXME.
      assert(0 && "fixupNeedsRelaxation() unimplemented");
      return false;
    }
    void relaxInstruction(const MCInst &Inst, MCInst &Res) const {
      // FIXME.
      assert(0 && "relaxInstruction() unimplemented");
    }

    bool writeNopData(uint64_t Count, MCObjectWriter *OW) const {
      // FIXME: Zero fill for now.
      for (uint64_t i = 0; i != Count; ++i)
        OW->Write8(0);
      return true;
    }

    bool is64Bit() const {
      StringRef name = TheTarget.getName();
      return name == "sparcv9";
    }
  };

  class ELFSparcAsmBackend : public SparcAsmBackend {
    Triple::OSType OSType;
  public:
    ELFSparcAsmBackend(const Target &T, Triple::OSType OSType) :
      SparcAsmBackend(T), OSType(OSType) { }

    void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                    uint64_t Value) const {

      Value = adjustFixupValue(Fixup.getKind(), Value);
      if (!Value) return;           // Doesn't change encoding.

      unsigned Offset = Fixup.getOffset();

      // For each byte of the fragment that the fixup touches, mask in the bits
      // from the fixup value. The Value has been "split up" into the
      // appropriate bitfields above.
      for (unsigned i = 0; i != 4; ++i)
        Data[Offset + i] |= uint8_t((Value >> ((4 - i - 1)*8)) & 0xff);

    }

    MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
      uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(OSType);
      return createSparcELFObjectWriter(OS, is64Bit(), OSABI);
    }

    virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
      return false;
    }
  };

} // end anonymous namespace


MCAsmBackend *llvm::createSparcAsmBackend(const Target &T,
                                          const MCRegisterInfo &MRI,
                                          StringRef TT,
                                          StringRef CPU) {
  return new ELFSparcAsmBackend(T, Triple(TT).getOS());
}
