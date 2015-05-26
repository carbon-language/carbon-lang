//===- lib/ReaderWriter/ELF/Mips/MipsReginfo.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_REGINFO_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_REGINFO_H

#include "llvm/Object/ELFTypes.h"

namespace llvm {
namespace object {

template <class ELFT>
struct Elf_RegInfo;

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_pad;         // unused padding field
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

template <class ELFT> struct Elf_Mips_Options {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  uint8_t kind;     // Determines interpretation of variable part of descriptor
  uint8_t size;     // Byte size of descriptor, including this header
  Elf_Half section; // Section header index of section affected,
                    // or 0 for global options
  Elf_Word info;    // Kind-specific information
};

} // end namespace object.
} // end namespace llvm.

namespace lld {
namespace elf {

struct MipsReginfo {
  uint32_t _gpRegMask = 0;
  uint32_t _cpRegMask[4];

  MipsReginfo() { memset(_cpRegMask, 0, sizeof(_cpRegMask)); }

  template <class ElfReginfo> MipsReginfo(const ElfReginfo &elf) {
    _gpRegMask = elf.ri_gprmask;
    _cpRegMask[0] = elf.ri_cprmask[0];
    _cpRegMask[1] = elf.ri_cprmask[1];
    _cpRegMask[2] = elf.ri_cprmask[2];
    _cpRegMask[3] = elf.ri_cprmask[3];
  }

  void merge(const MipsReginfo &info) {
    _gpRegMask |= info._gpRegMask;
    _cpRegMask[0] |= info._cpRegMask[0];
    _cpRegMask[1] |= info._cpRegMask[1];
    _cpRegMask[2] |= info._cpRegMask[2];
    _cpRegMask[3] |= info._cpRegMask[3];
  }
};

} // end namespace elf
} // end namespace lld

#endif
