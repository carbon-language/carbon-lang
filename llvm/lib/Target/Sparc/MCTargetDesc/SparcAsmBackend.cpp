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
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {
  class SparcAsmBackend : public MCAsmBackend {

  public:
    SparcAsmBackend(const Target &T) : MCAsmBackend() {}

    unsigned getNumFixupKinds() const {
      return Sparc::NumTargetFixupKinds;
    }

    const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
      const static MCFixupKindInfo Infos[Sparc::NumTargetFixupKinds] = {
        // name                    offset bits  flags
        { "fixup_sparc_call30",     0,     30,  MCFixupKindInfo::FKF_IsPCRel },
        { "fixup_sparc_br22",       0,     22,  MCFixupKindInfo::FKF_IsPCRel },
        { "fixup_sparc_br19",       0,     19,  MCFixupKindInfo::FKF_IsPCRel }
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
  };

  class ELFSparcAsmBackend : public SparcAsmBackend {
  public:
    ELFSparcAsmBackend(const Target &T, Triple::OSType OSType) :
      SparcAsmBackend(T) { }

    void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                    uint64_t Value) const {
      assert(0 && "applyFixup not implemented yet");
    }

    MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
      assert(0 && "Object Writer not implemented yet");
      return 0;
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
