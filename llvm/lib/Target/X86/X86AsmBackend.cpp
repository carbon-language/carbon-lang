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
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MachObjectWriter.h"
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
};

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
