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
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default: assert(0 && "invalid fixup kind!");
  case FK_PCRel_1:
  case FK_Data_1: return 0;
  case FK_PCRel_2:
  case FK_Data_2: return 1;
  case FK_PCRel_4:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
  case X86::reloc_signed_4byte:
  case X86::reloc_global_offset_table:
  case FK_Data_4: return 2;
  case FK_Data_8: return 3;
  }
}

namespace {
class X86MachObjectWriter : public MCMachObjectTargetWriter {
public:
  X86MachObjectWriter(bool Is64Bit, uint32_t CPUType,
                      uint32_t CPUSubtype)
    : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype,
                               /*UseAggressiveSymbolFolding=*/Is64Bit) {}
};

class X86ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  X86ELFObjectWriter(bool is64Bit, Triple::OSType OSType, uint16_t EMachine,
                     bool HasRelocationAddend)
    : MCELFObjectTargetWriter(is64Bit, OSType, EMachine, HasRelocationAddend) {}
};

class X86AsmBackend : public TargetAsmBackend {
public:
  X86AsmBackend(const Target &T)
    : TargetAsmBackend() {}

  unsigned getNumFixupKinds() const {
    return X86::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[X86::NumTargetFixupKinds] = {
      { "reloc_riprel_4byte", 0, 4 * 8, MCFixupKindInfo::FKF_IsPCRel },
      { "reloc_riprel_4byte_movq_load", 0, 4 * 8, MCFixupKindInfo::FKF_IsPCRel},
      { "reloc_signed_4byte", 0, 4 * 8, 0},
      { "reloc_global_offset_table", 0, 4 * 8, 0}
    };

    if (Kind < FirstTargetFixupKind)
      return TargetAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const {
    unsigned Size = 1 << getFixupKindLog2Size(Fixup.getKind());

    assert(Fixup.getOffset() + Size <= DataSize &&
           "Invalid fixup offset!");
    for (unsigned i = 0; i != Size; ++i)
      Data[Fixup.getOffset() + i] = uint8_t(Value >> (i * 8));
  }

  bool MayNeedRelaxation(const MCInst &Inst) const;

  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const;

  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const;
};
} // end anonymous namespace

static unsigned getRelaxedOpcodeBranch(unsigned Op) {
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

static unsigned getRelaxedOpcodeArith(unsigned Op) {
  switch (Op) {
  default:
    return Op;

    // IMUL
  case X86::IMUL16rri8: return X86::IMUL16rri;
  case X86::IMUL16rmi8: return X86::IMUL16rmi;
  case X86::IMUL32rri8: return X86::IMUL32rri;
  case X86::IMUL32rmi8: return X86::IMUL32rmi;
  case X86::IMUL64rri8: return X86::IMUL64rri32;
  case X86::IMUL64rmi8: return X86::IMUL64rmi32;

    // AND
  case X86::AND16ri8: return X86::AND16ri;
  case X86::AND16mi8: return X86::AND16mi;
  case X86::AND32ri8: return X86::AND32ri;
  case X86::AND32mi8: return X86::AND32mi;
  case X86::AND64ri8: return X86::AND64ri32;
  case X86::AND64mi8: return X86::AND64mi32;

    // OR
  case X86::OR16ri8: return X86::OR16ri;
  case X86::OR16mi8: return X86::OR16mi;
  case X86::OR32ri8: return X86::OR32ri;
  case X86::OR32mi8: return X86::OR32mi;
  case X86::OR64ri8: return X86::OR64ri32;
  case X86::OR64mi8: return X86::OR64mi32;

    // XOR
  case X86::XOR16ri8: return X86::XOR16ri;
  case X86::XOR16mi8: return X86::XOR16mi;
  case X86::XOR32ri8: return X86::XOR32ri;
  case X86::XOR32mi8: return X86::XOR32mi;
  case X86::XOR64ri8: return X86::XOR64ri32;
  case X86::XOR64mi8: return X86::XOR64mi32;

    // ADD
  case X86::ADD16ri8: return X86::ADD16ri;
  case X86::ADD16mi8: return X86::ADD16mi;
  case X86::ADD32ri8: return X86::ADD32ri;
  case X86::ADD32mi8: return X86::ADD32mi;
  case X86::ADD64ri8: return X86::ADD64ri32;
  case X86::ADD64mi8: return X86::ADD64mi32;

    // SUB
  case X86::SUB16ri8: return X86::SUB16ri;
  case X86::SUB16mi8: return X86::SUB16mi;
  case X86::SUB32ri8: return X86::SUB32ri;
  case X86::SUB32mi8: return X86::SUB32mi;
  case X86::SUB64ri8: return X86::SUB64ri32;
  case X86::SUB64mi8: return X86::SUB64mi32;

    // CMP
  case X86::CMP16ri8: return X86::CMP16ri;
  case X86::CMP16mi8: return X86::CMP16mi;
  case X86::CMP32ri8: return X86::CMP32ri;
  case X86::CMP32mi8: return X86::CMP32mi;
  case X86::CMP64ri8: return X86::CMP64ri32;
  case X86::CMP64mi8: return X86::CMP64mi32;

    // PUSH
  case X86::PUSHi8: return X86::PUSHi32;
  }
}

static unsigned getRelaxedOpcode(unsigned Op) {
  unsigned R = getRelaxedOpcodeArith(Op);
  if (R != Op)
    return R;
  return getRelaxedOpcodeBranch(Op);
}

bool X86AsmBackend::MayNeedRelaxation(const MCInst &Inst) const {
  // Branches can always be relaxed.
  if (getRelaxedOpcodeBranch(Inst.getOpcode()) != Inst.getOpcode())
    return true;

  // Check if this instruction is ever relaxable.
  if (getRelaxedOpcodeArith(Inst.getOpcode()) == Inst.getOpcode())
    return false;


  // Check if it has an expression and is not RIP relative.
  bool hasExp = false;
  bool hasRIP = false;
  for (unsigned i = 0; i < Inst.getNumOperands(); ++i) {
    const MCOperand &Op = Inst.getOperand(i);
    if (Op.isExpr())
      hasExp = true;

    if (Op.isReg() && Op.getReg() == X86::RIP)
      hasRIP = true;
  }

  // FIXME: Why exactly do we need the !hasRIP? Is it just a limitation on
  // how we do relaxations?
  return hasExp && !hasRIP;
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
bool X86AsmBackend::WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
  static const uint8_t Nops[10][10] = {
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
  };

  // Write an optimal sequence for the first 15 bytes.
  const uint64_t OptimalCount = (Count < 16) ? Count : 15;
  const uint64_t Prefixes = OptimalCount <= 10 ? 0 : OptimalCount - 10;
  for (uint64_t i = 0, e = Prefixes; i != e; i++)
    OW->Write8(0x66);
  const uint64_t Rest = OptimalCount - Prefixes;
  for (uint64_t i = 0, e = Rest; i != e; i++)
    OW->Write8(Nops[Rest - 1][i]);

  // Finish with single byte nops.
  for (uint64_t i = OptimalCount, e = Count; i != e; ++i)
   OW->Write8(0x90);

  return true;
}

/* *** */

namespace {
class ELFX86AsmBackend : public X86AsmBackend {
  MCELFObjectFormat Format;

public:
  Triple::OSType OSType;
  ELFX86AsmBackend(const Target &T, Triple::OSType _OSType)
    : X86AsmBackend(T), OSType(_OSType) {
    HasReliableSymbolDifference = true;
  }

  virtual const MCObjectFormat &getObjectFormat() const {
    return Format;
  }

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    const MCSectionELF &ES = static_cast<const MCSectionELF&>(Section);
    return ES.getFlags() & MCSectionELF::SHF_MERGE;
  }
};

class ELFX86_32AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_32AsmBackend(const Target &T, Triple::OSType OSType)
    : ELFX86AsmBackend(T, OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(new X86ELFObjectWriter(false, OSType,
                                                        ELF::EM_386, false),
                                 OS, /*IsLittleEndian*/ true);
  }
};

class ELFX86_64AsmBackend : public ELFX86AsmBackend {
public:
  ELFX86_64AsmBackend(const Target &T, Triple::OSType OSType)
    : ELFX86AsmBackend(T, OSType) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(new X86ELFObjectWriter(true, OSType,
                                                        ELF::EM_X86_64, true),
                                 OS, /*IsLittleEndian*/ true);
  }
};

class WindowsX86AsmBackend : public X86AsmBackend {
  bool Is64Bit;
  MCCOFFObjectFormat Format;

public:
  WindowsX86AsmBackend(const Target &T, bool is64Bit)
    : X86AsmBackend(T)
    , Is64Bit(is64Bit) {
  }

  virtual const MCObjectFormat &getObjectFormat() const {
    return Format;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createWinCOFFObjectWriter(OS, Is64Bit);
  }
};

class DarwinX86AsmBackend : public X86AsmBackend {
  MCMachOObjectFormat Format;

public:
  DarwinX86AsmBackend(const Target &T)
    : X86AsmBackend(T) { }

  virtual const MCObjectFormat &getObjectFormat() const {
    return Format;
  }
};

class DarwinX86_32AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_32AsmBackend(const Target &T)
    : DarwinX86AsmBackend(T) {}

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createMachObjectWriter(new X86MachObjectWriter(
                                    /*Is64Bit=*/false,
                                    object::mach::CTM_i386,
                                    object::mach::CSX86_ALL),
                                  OS, /*IsLittleEndian=*/true);
  }
};

class DarwinX86_64AsmBackend : public DarwinX86AsmBackend {
public:
  DarwinX86_64AsmBackend(const Target &T)
    : DarwinX86AsmBackend(T) {
    HasReliableSymbolDifference = true;
  }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createMachObjectWriter(new X86MachObjectWriter(
                                    /*Is64Bit=*/true,
                                    object::mach::CTM_x86_64,
                                    object::mach::CSX86_ALL),
                                  OS, /*IsLittleEndian=*/true);
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
    return new ELFX86_32AsmBackend(T, Triple(TT).getOS());
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
    return new ELFX86_64AsmBackend(T, Triple(TT).getOS());
  }
}
