//===- lib/ReaderWriter/ELF/AArch64/AArch64ELFFile.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class AArch64LinkingContext;

template <class ELFT> class AArch64ELFFile : public ELFFile<ELFT> {
public:
  AArch64ELFFile(std::unique_ptr<MemoryBuffer> mb, AArch64LinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<AArch64ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, AArch64LinkingContext &ctx) {
    return std::unique_ptr<AArch64ELFFile<ELFT>>(
        new AArch64ELFFile<ELFT>(std::move(mb), ctx));
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H
