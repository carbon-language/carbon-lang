//===-- HexagonAsmBackend.cpp - Hexagon Assembler Backend -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonFixupKinds.h"
#include "HexagonMCTargetDesc.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"

using namespace llvm;
using namespace Hexagon;

namespace {

class HexagonAsmBackend : public MCAsmBackend {
  mutable uint64_t relaxedCnt;
  std::unique_ptr <MCInstrInfo> MCII;
  std::unique_ptr <MCInst *> RelaxTarget;
public:
  HexagonAsmBackend(Target const & /*T*/) :
    MCII (createHexagonMCInstrInfo()), RelaxTarget(new MCInst *){}

  unsigned getNumFixupKinds() const override { return 0; }

  void applyFixup(MCFixup const & /*Fixup*/, char * /*Data*/,
                  unsigned /*DataSize*/, uint64_t /*Value*/,
                  bool /*IsPCRel*/) const override {
    return;
  }

  bool isInstRelaxable(MCInst const &HMI) const {
    const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(*MCII, HMI);
    bool Relaxable = false;
    // Branches and loop-setup insns are handled as necessary by relaxation.
    if (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeJ ||
        (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeNV &&
         MCID.isBranch()) ||
        (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeCR &&
         HMI.getOpcode() != Hexagon::C4_addipc))
      if (HexagonMCInstrInfo::isExtendable(*MCII, HMI))
        Relaxable = true;

    return Relaxable;
  }

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \param Inst - The instruction to test.
  bool mayNeedRelaxation(MCInst const &Inst) const override {
    assert(HexagonMCInstrInfo::isBundle(Inst));
    bool PreviousIsExtender = false;
    for (auto const &I : HexagonMCInstrInfo::bundleInstructions(Inst)) {
      auto const &Inst = *I.getInst();
      if (!PreviousIsExtender) {
        if (isInstRelaxable(Inst))
          return true;
      }
      PreviousIsExtender = HexagonMCInstrInfo::isImmext(Inst);
    }
    return false;
  }

  /// fixupNeedsRelaxation - Target specific predicate for whether a given
  /// fixup requires the associated instruction to be relaxed.
  bool fixupNeedsRelaxationAdvanced(const MCFixup &Fixup, bool Resolved,
                                    uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout) const override {
    MCInst const &MCB = DF->getInst();
    assert(HexagonMCInstrInfo::isBundle(MCB));

    *RelaxTarget = nullptr;
    MCInst &MCI = const_cast<MCInst &>(HexagonMCInstrInfo::instruction(
        MCB, Fixup.getOffset() / HEXAGON_INSTR_SIZE));
    // If we cannot resolve the fixup value, it requires relaxation.
    if (!Resolved) {
      switch ((unsigned)Fixup.getKind()) {
      case fixup_Hexagon_B22_PCREL:
      // GetFixupCount assumes B22 won't relax
      // Fallthrough
      default:
        return false;
        break;
      case fixup_Hexagon_B13_PCREL:
      case fixup_Hexagon_B15_PCREL:
      case fixup_Hexagon_B9_PCREL:
      case fixup_Hexagon_B7_PCREL: {
        if (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_SIZE) {
          ++relaxedCnt;
          *RelaxTarget = &MCI;
          return true;
        } else {
          return false;
        }
        break;
      }
      }
    }
    bool Relaxable = isInstRelaxable(MCI);
    if (Relaxable == false)
      return false;

    MCFixupKind Kind = Fixup.getKind();
    int64_t sValue = Value;
    int64_t maxValue;

    switch ((unsigned)Kind) {
    case fixup_Hexagon_B7_PCREL:
      maxValue = 1 << 8;
      break;
    case fixup_Hexagon_B9_PCREL:
      maxValue = 1 << 10;
      break;
    case fixup_Hexagon_B15_PCREL:
      maxValue = 1 << 16;
      break;
    case fixup_Hexagon_B22_PCREL:
      maxValue = 1 << 23;
      break;
    default:
      maxValue = INT64_MAX;
      break;
    }

    bool isFarAway = -maxValue > sValue || sValue > maxValue - 1;

    if (isFarAway) {
      if (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_SIZE) {
        ++relaxedCnt;
        *RelaxTarget = &MCI;
        return true;
      }
    }

    return false;
  }

  /// Simple predicate for targets where !Resolved implies requiring relaxation
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    llvm_unreachable("Handled by fixupNeedsRelaxationAdvanced");
  }

  void relaxInstruction(MCInst const & /*Inst*/,
                        MCInst & /*Res*/) const override {
    llvm_unreachable("relaxInstruction() unimplemented");
  }

  bool writeNopData(uint64_t /*Count*/,
                    MCObjectWriter * /*OW*/) const override {
    return true;
  }
};
} // end anonymous namespace

namespace {
class ELFHexagonAsmBackend : public HexagonAsmBackend {
  uint8_t OSABI;

public:
  ELFHexagonAsmBackend(Target const &T, uint8_t OSABI)
      : HexagonAsmBackend(T), OSABI(OSABI) {}

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
    StringRef CPU("HexagonV4");
    return createHexagonELFObjectWriter(OS, OSABI, CPU);
  }
};
} // end anonymous namespace

namespace llvm {
MCAsmBackend *createHexagonAsmBackend(Target const &T,
                                      MCRegisterInfo const & /*MRI*/,
                                      StringRef TT, StringRef /*CPU*/) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(Triple(TT).getOS());
  return new ELFHexagonAsmBackend(T, OSABI);
}
}
