//===-- ARMWinCOFFObjectWriter.cpp - ARM Windows COFF Object Writer -- C++ -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMFixupKinds.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWinCOFFObjectWriter.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {
class ARMWinCOFFObjectWriter : public MCWinCOFFObjectTargetWriter {
public:
  ARMWinCOFFObjectWriter(bool Is64Bit)
    : MCWinCOFFObjectTargetWriter(COFF::IMAGE_FILE_MACHINE_ARMNT) {
    assert(!Is64Bit && "AArch64 support not yet implemented");
  }
  virtual ~ARMWinCOFFObjectWriter() { }

  unsigned getRelocType(const MCValue &Target, const MCFixup &Fixup,
                        bool IsCrossSection) const override;
};

unsigned ARMWinCOFFObjectWriter::getRelocType(const MCValue &Target,
                                              const MCFixup &Fixup,
                                              bool IsCrossSection) const {
  assert(getMachine() == COFF::IMAGE_FILE_MACHINE_ARMNT &&
         "AArch64 support not yet implemented");

  MCSymbolRefExpr::VariantKind Modifier =
    Target.isAbsolute() ? MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  switch (static_cast<unsigned>(Fixup.getKind())) {
  default: llvm_unreachable("unsupported relocation type");
  case FK_Data_4:
    switch (Modifier) {
    case MCSymbolRefExpr::VK_COFF_IMGREL32:
      return COFF::IMAGE_REL_ARM_ADDR32NB;
    case MCSymbolRefExpr::VK_SECREL:
      return COFF::IMAGE_REL_ARM_SECREL;
    default:
      return COFF::IMAGE_REL_ARM_ADDR32;
    }
  case ARM::fixup_t2_condbranch:
    return COFF::IMAGE_REL_ARM_BRANCH20T;
  case ARM::fixup_t2_uncondbranch:
    return COFF::IMAGE_REL_ARM_BRANCH24T;
  case ARM::fixup_arm_thumb_bl:
  case ARM::fixup_arm_thumb_blx:
    return COFF::IMAGE_REL_ARM_BLX23T;
  case ARM::fixup_t2_movw_lo16:
    return COFF::IMAGE_REL_ARM_MOV32T;
  case ARM::fixup_t2_movt_hi16:
    llvm_unreachable("High-word for pair-wise relocations are contiguously "
                     "addressed as an IMAGE_REL_ARM_MOV32T relocation");
  }
}
}

namespace llvm {
MCObjectWriter *createARMWinCOFFObjectWriter(raw_ostream &OS, bool Is64Bit) {
  MCWinCOFFObjectTargetWriter *MOTW = new ARMWinCOFFObjectWriter(Is64Bit);
  return createWinCOFFObjectWriter(MOTW, OS);
}
}

