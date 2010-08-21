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
#include "llvm/MC/ELFObjectWriter.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MachObjectWriter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;


static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default: assert(0 && "invalid fixup kind!");
  case X86::reloc_pcrel_1byte:
  case FK_Data_1: return 0;
  case X86::reloc_pcrel_2byte:
  case FK_Data_2: return 1;
  case X86::reloc_pcrel_4byte:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case FK_Data_4: return 2;
  case FK_Data_8: return 3;
  }
}

namespace {
class X86AsmBackend : public TargetAsmBackend {
public:
  X86AsmBackend(const Target &T)
    : TargetAsmBackend(T) {}

  void ApplyFixup(const MCFixup &Fixup, MCDataFragment &DF,
                  uint64_t Value) const {
    unsigned Size = 1 << getFixupKindLog2Size(Fixup.getKind());

    assert(Fixup.getOffset() + Size <= DF.getContents().size() &&
           "Invalid fixup offset!");
    for (unsigned i = 0; i != Size; ++i)
      DF.getContents()[Fixup.getOffset() + i] = uint8_t(Value >> (i * 8));
  }

  bool MayNeedRelaxation(const MCInst &Inst) const;

  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const;

  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const;
};
} // end anonymous namespace 

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

bool X86AsmBackend::MayNeedRelaxation(const MCInst &Inst) const {
  // Check if this instruction is ever relaxable.
  if (getRelaxedOpcode(Inst.getOpcode()) == Inst.getOpcode())
    return false;

  // If so, just assume it can be relaxed. Once we support relaxing more complex
  // instructions we should check that the instruction actually has symbolic
  // operands before doing this, but we need to be careful about things like
  // PCrel.
  return true;
}

// FIXME: Can tblgen help at all here to verify there aren't other instructions
// we can relax?
void X86AsmBackend::RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
  // The only relaxations X86 does is from a 1byte pcrel to a 4byte pcrel.
  unsigned RelaxedOp = getRelaxedOpcode(Inst.getOpcode());

  if (RelaxedOp == Inst.getOpcode()) {
    SmallString<256> Tmp;
    raw_svector_ostream OS(Tmp);
    Inst.dump_pretty(OS);
    OS << "\n";
    report_fatal_error("unexpected instruction to relax: " + OS.str());
  }

  Res = Inst;
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

namespace {
class ELFX86AsmBackend : public X86AsmBackend {
public:
  ELFX86AsmBackend(const Target &T)
    : X86AsmBackend(T) {
    HasAbsolutizedSet = true;
    HasScatteredSymbols = true;
  }

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionELF &SE = static_cast<const MCSectionELF&>(Section);
    return SE.getType() == MCSectionELF::SHT_NOBITS;;
  }
};

class ELFX86_32AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_32AsmBackend(const Target &T)
    : ELFX86AsmBackend(T) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return new ELFObjectWriter(OS, /*Is64Bit=*/false,
                               /*IsLittleEndian=*/true,
                               /*HasRelocationAddend=*/false);
  }
};

class ELFX86_64AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_64AsmBackend(const Target &T)
    : ELFX86AsmBackend(T) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return new ELFObjectWriter(OS, /*Is64Bit=*/true,
                               /*IsLittleEndian=*/true,
                               /*HasRelocationAddend=*/true);
  }
};

class WindowsX86AsmBackend : public X86AsmBackend {
  bool Is64Bit;
public:
  WindowsX86AsmBackend(const Target &T, bool is64Bit)
    : X86AsmBackend(T)
    , Is64Bit(is64Bit) {
    HasScatteredSymbols = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createWinCOFFObjectWriter(OS, Is64Bit);
  }

  bool isVirtualSection(const MCSection &Section) const {
    const MCSectionCOFF &SE = static_cast<const MCSectionCOFF&>(Section);
    return SE.getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
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
            SMO.getType() == MCSectionMachO::S_GB_ZEROFILL ||
            SMO.getType() == MCSectionMachO::S_THREAD_LOCAL_ZEROFILL);
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

  virtual bool isSectionAtomizable(const MCSection &Section) const {
    const MCSectionMachO &SMO = static_cast<const MCSectionMachO&>(Section);
    // Fixed sized data sections are uniqued, they cannot be diced into atoms.
    switch (SMO.getType()) {
    default:
      return true;

    case MCSectionMachO::S_4BYTE_LITERALS:
    case MCSectionMachO::S_8BYTE_LITERALS:
    case MCSectionMachO::S_16BYTE_LITERALS:
    case MCSectionMachO::S_LITERAL_POINTERS:
    case MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS:
    case MCSectionMachO::S_LAZY_SYMBOL_POINTERS:
    case MCSectionMachO::S_MOD_INIT_FUNC_POINTERS:
    case MCSectionMachO::S_MOD_TERM_FUNC_POINTERS:
    case MCSectionMachO::S_INTERPOSING:
      return false;
    }
  }
};

} // end anonymous namespace 

TargetAsmBackend *llvm::createX86_32AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86_32AsmBackend(T);
  case Triple::MinGW32:
  case Triple::Cygwin:
  case Triple::Win32:
    return new WindowsX86AsmBackend(T, false);
  default:
    return new ELFX86_32AsmBackend(T);
  }
}

TargetAsmBackend *llvm::createX86_64AsmBackend(const Target &T,
                                               const std::string &TT) {
  switch (Triple(TT).getOS()) {
  case Triple::Darwin:
    return new DarwinX86_64AsmBackend(T);
  case Triple::MinGW64:
  case Triple::Cygwin:
  case Triple::Win32:
    return new WindowsX86AsmBackend(T, true);
  default:
    return new ELFX86_64AsmBackend(T);
  }
}
