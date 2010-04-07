//===-- X86AsmBackend.cpp - X86 Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmBackend.h"
#include "X86.h"
#include "X86FixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MachObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

namespace {

static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default: assert(0 && "invalid fixup kind!");
  case X86::reloc_pcrel_1byte:
  case FK_Data_1: return 0;
  case FK_Data_2: return 1;
  case X86::reloc_pcrel_4byte:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case FK_Data_4: return 2;
  case FK_Data_8: return 3;
  }
}

class X86AsmBackend : public TargetAsmBackend {
public:
  X86AsmBackend(const Target &T)
    : TargetAsmBackend(T) {}

  void ApplyFixup(const MCAsmFixup &Fixup, MCDataFragment &DF,
                  uint64_t Value) const {
    unsigned Size = 1 << getFixupKindLog2Size(Fixup.Kind);

    assert(Fixup.Offset + Size <= DF.getContents().size() &&
           "Invalid fixup offset!");
    for (unsigned i = 0; i != Size; ++i)
      DF.getContents()[Fixup.Offset + i] = uint8_t(Value >> (i * 8));
  }

  bool MayNeedRelaxation(const MCInst &Inst,
                         const SmallVectorImpl<MCAsmFixup> &Fixups) const;

  void RelaxInstruction(const MCInstFragment *IF, MCInst &Res) const;

  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const;
};

static unsigned getRelaxedOpcode(unsigned Op) {
  switch (Op) {
  default:
    return Op;

  case X86::JAE_1: return X86::JAE_4;
  case X86::JA_1:  return X86::JA_4;
  case X86::JBE_1: return X86::JBE_4;
  case X86::JB_1:  return X86::JB_4;
  case X86::JE_1:  return X86::JE_4;
  case X86::JGE_1: return X86::JGE_4;
  case X86::JG_1:  return X86::JG_4;
  case X86::JLE_1: return X86::JLE_4;
  case X86::JL_1:  return X86::JL_4;
  case X86::JMP_1: return X86::JMP_4;
  case X86::JNE_1: return X86::JNE_4;
  case X86::JNO_1: return X86::JNO_4;
  case X86::JNP_1: return X86::JNP_4;
  case X86::JNS_1: return X86::JNS_4;
  case X86::JO_1:  return X86::JO_4;
  case X86::JP_1:  return X86::JP_4;
  case X86::JS_1:  return X86::JS_4;
  }
}

bool X86AsmBackend::MayNeedRelaxation(const MCInst &Inst,
                              const SmallVectorImpl<MCAsmFixup> &Fixups) const {
  // Check for a 1byte pcrel fixup, and enforce that we would know how to relax
  // this instruction.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    if (unsigned(Fixups[i].Kind) == X86::reloc_pcrel_1byte) {
      assert(getRelaxedOpcode(Inst.getOpcode()) != Inst.getOpcode());
      return true;
    }
  }

  return false;
}

// FIXME: Can tblgen help at all here to verify there aren't other instructions
// we can relax?
void X86AsmBackend::RelaxInstruction(const MCInstFragment *IF,
                                     MCInst &Res) const {
  // The only relaxations X86 does is from a 1byte pcrel to a 4byte pcrel.
  unsigned RelaxedOp = getRelaxedOpcode(IF->getInst().getOpcode());

  if (RelaxedOp == IF->getInst().getOpcode()) {
    SmallString<256> Tmp;
    raw_svector_ostream OS(Tmp);
    IF->getInst().dump_pretty(OS);
    report_fatal_error("unexpected instruction to relax: " + OS.str());
  }

  Res = IF->getInst();
  Res.setOpcode(RelaxedOp);
}

/// WriteNopData - Write optimal nops to the output file for the \arg Count
/// bytes.  This returns the number of bytes written.  It may return 0 if
/// the \arg Count is more than the maximum optimal nops.
///
/// FIXME this is X86 32-bit specific and should move to a better place.
bool X86AsmBackend::WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
  static const uint8_t Nops[16][16] = {
    // nop
    {0x90},
    // xchg %ax,%ax
    {0x66, 0x90},
    // nopl (%[re]ax)
    {0x0f, 0x1f, 0x00},
    // nopl 0(%[re]ax)
    {0x0f, 0x1f, 0x40, 0x00},
    // nopl 0(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw 0L(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopw %cs:0L(%[re]ax,%[re]ax,1)
    {0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    // nopl 0(%[re]ax,%[re]ax,1)
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    // nopw 0(%[re]ax,%[re]ax,1)
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
    // nopw 0(%[re]ax,%[re]ax,1)
    // nopl 0L(%[re]ax) */
    {0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
     0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    // nopl 0L(%[re]ax)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
     0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00},
    // nopl 0L(%[re]ax)
    // nopl 0L(%[re]ax,%[re]ax,1)
    {0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
     0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00}
  };

  // Write an optimal sequence for the first 15 bytes.
  uint64_t OptimalCount = (Count < 16) ? Count : 15;
  for (uint64_t i = 0, e = OptimalCount; i != e; i++)
    OW->Write8(Nops[OptimalCount - 1][i]);

  // Finish with single byte nops.
  for (uint64_t i = OptimalCount, e = Count; i != e; ++i)
   OW->Write8(0x90);

  return true;
}

/* *** */

class ELFX86AsmBackend : public X86AsmBackend {
public:
  ELFX86AsmBackend(const Target &T)
    : X86AsmBackend(T) {
    HasAbsolutizedSet = true;
    HasScatteredSymbols = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return 0;
  }

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionELF &SE = static_cast<const MCSectionELF&>(Section);
    return SE.getType() == MCSectionELF::SHT_NOBITS;;
  }
};

class DarwinX86AsmBackend : public X86AsmBackend {
public:
  DarwinX86AsmBackend(const Target &T)
    : X86AsmBackend(T) {
    HasAbsolutizedSet = true;
    HasScatteredSymbols = true;
  }

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    return (SMO.getType() == MCSectionMachO::S_ZEROFILL ||
            SMO.getType() == MCSectionMachO::S_GB_ZEROFILL);
  }
};

class DarwinX86_32AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_32AsmBackend(const Target &T)
    : DarwinX86AsmBackend(T) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return new MachObjectWriter(OS, /*Is64Bit=*/false);
  }
};

class DarwinX86_64AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_64AsmBackend(const Target &T)
    : DarwinX86AsmBackend(T) {
    HasReliableSymbolDifference = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return new MachObjectWriter(OS, /*Is64Bit=*/true);
  }

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    // Temporary labels in the string literals sections require symbols. The
    // issue is that the x86_64 relocation format does not allow symbol +
    // offset, and so the linker does not have enough information to resolve the
    // access to the appropriate atom unless an external relocation is used. For
    // non-cstring sections, we expect the compiler to use a non-temporary label
    // for anything that could have an addend pointing outside the symbol.
    //
    // See <rdar://problem/4765733>.
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    return SMO.getType() == MCSectionMachO::S_CSTRING_LITERALS;
  }
};

}

TargetAsmBackend *llvm::createX86_32AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86_32AsmBackend(T);
  default:
    return new ELFX86AsmBackend(T);
  }
}

TargetAsmBackend *llvm::createX86_64AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86_64AsmBackend(T);
  default:
    return new ELFX86AsmBackend(T);
  }
}
