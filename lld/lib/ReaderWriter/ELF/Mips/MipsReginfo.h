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
  LLVM_ELF_IMPORT_TYPES(ELFT::TargetEndianness, ELFT::MaxAlignment,
                        ELFT::Is64Bits)
  uint8_t kind;     // Determines interpretation of variable part of descriptor
  uint8_t size;     // Byte size of descriptor, including this header
  Elf_Half section; // Section header index of section affected,
                    // or 0 for global options
  Elf_Word info;    // Kind-specific information
};

} // end namespace object.
} // end namespace llvm.

#endif
