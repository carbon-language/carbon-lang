//===- lib/ReaderWriter/ELF/FileCommon.h ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_FILE_COMMON_H
#define LLD_READER_WRITER_ELF_FILE_COMMON_H

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

template <class ELFT>
std::error_code checkCompatibility(unsigned char size, unsigned char endian);

template <typename ELFT>
std::error_code isCompatible(MemoryBufferRef mb, ELFLinkingContext &ctx) {
  typedef llvm::object::Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  if (uintptr_t(mb.getBufferStart()) & 1)
    return make_dynamic_error_code("invalid alignment");

  auto *hdr = reinterpret_cast<const Elf_Ehdr *>(mb.getBuffer().data());
  if (hdr->e_machine != ctx.getMachineType())
    return make_dynamic_error_code("incompatible machine type");

  unsigned char size;
  unsigned char endian;
  std::tie(size, endian) = llvm::object::getElfArchType(mb.getBuffer());
  if (std::error_code ec = checkCompatibility<ELFT>(size, endian))
    return ec;

  if (auto ec = ctx.mergeHeaderFlags(hdr->getFileClass(), hdr->e_flags))
    return ec;
  return std::error_code();
}

} // end namespace elf
} // end namespace lld

#endif
