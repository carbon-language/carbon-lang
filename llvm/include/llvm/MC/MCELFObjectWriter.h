//===-- llvm/MC/MCELFObjectWriter.h - ELF Object Writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFOBJECTWRITER_H
#define LLVM_MC_MCELFOBJECTWRITER_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ELF.h"
#include <vector>

namespace llvm {
class MCAssembler;
class MCFixup;
class MCFragment;
class MCObjectWriter;
class MCSymbol;
class MCValue;

/// @name Relocation Data
/// @{

struct ELFRelocationEntry {
  // Make these big enough for both 32-bit and 64-bit
  uint64_t r_offset;
  int Index;
  unsigned Type;
  const MCSymbol *Symbol;
  uint64_t r_addend;
  const MCFixup *Fixup;

  ELFRelocationEntry()
    : r_offset(0), Index(0), Type(0), Symbol(0), r_addend(0), Fixup(0) {}

  ELFRelocationEntry(uint64_t RelocOffset, int Idx, unsigned RelType,
                     const MCSymbol *Sym, uint64_t Addend, const MCFixup &Fixup)
    : r_offset(RelocOffset), Index(Idx), Type(RelType), Symbol(Sym),
      r_addend(Addend), Fixup(&Fixup) {}

  // Support lexicographic sorting.
  bool operator<(const ELFRelocationEntry &RE) const {
    return RE.r_offset < r_offset;
  }
};

class MCELFObjectTargetWriter {
  const uint8_t OSABI;
  const uint16_t EMachine;
  const unsigned HasRelocationAddend : 1;
  const unsigned Is64Bit : 1;

protected:

  MCELFObjectTargetWriter(bool Is64Bit_, uint8_t OSABI_,
                          uint16_t EMachine_,  bool HasRelocationAddend_);

public:
  static uint8_t getOSABI(Triple::OSType OSType) {
    switch (OSType) {
      case Triple::FreeBSD:
        return ELF::ELFOSABI_FREEBSD;
      case Triple::Linux:
        return ELF::ELFOSABI_LINUX;
      default:
        return ELF::ELFOSABI_NONE;
    }
  }

  virtual ~MCELFObjectTargetWriter() {}

  virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                bool IsPCRel, bool IsRelocWithSymbol,
                                int64_t Addend) const = 0;
  virtual unsigned getEFlags() const;
  virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                         const MCValue &Target,
                                         const MCFragment &F,
                                         const MCFixup &Fixup,
                                         bool IsPCRel) const;
  virtual void adjustFixupOffset(const MCFixup &Fixup,
                                 uint64_t &RelocOffset);

  virtual void sortRelocs(const MCAssembler &Asm,
                          std::vector<ELFRelocationEntry> &Relocs);

  /// @name Accessors
  /// @{
  uint8_t getOSABI() { return OSABI; }
  uint16_t getEMachine() { return EMachine; }
  bool hasRelocationAddend() { return HasRelocationAddend; }
  bool is64Bit() const { return Is64Bit; }
  /// @}
};

/// \brief Construct a new ELF writer instance.
///
/// \param MOTW - The target specific ELF writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                      raw_ostream &OS, bool IsLittleEndian);
} // End llvm namespace

#endif
