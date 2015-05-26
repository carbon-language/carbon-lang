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
