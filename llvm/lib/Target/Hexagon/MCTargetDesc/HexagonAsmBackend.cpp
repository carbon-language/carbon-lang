//===-- HexagonAsmBackend.cpp - Hexagon Assembler Backend -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCELFObjectWriter.h"

using namespace llvm;

namespace {

class HexagonAsmBackend : public MCAsmBackend {
public:
  HexagonAsmBackend(Target const & /*T*/) {}

  unsigned getNumFixupKinds() const override { return 0; }

  void applyFixup(MCFixup const & /*Fixup*/, char * /*Data*/,
                  unsigned /*DataSize*/, uint64_t /*Value*/,
                  bool /*IsPCRel*/) const override {
    return;
  }

  bool mayNeedRelaxation(MCInst const & /*Inst*/) const override {
    return false;
  }

  bool fixupNeedsRelaxation(MCFixup const & /*Fixup*/, uint64_t /*Value*/,
                            MCRelaxableFragment const * /*DF*/,
                            MCAsmLayout const & /*Layout*/) const override {
    llvm_unreachable("fixupNeedsRelaxation() unimplemented");
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

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const override {
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
