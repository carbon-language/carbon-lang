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
  AArch64ELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings) {}

  static ErrorOr<std::unique_ptr<AArch64ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    std::unique_ptr<AArch64ELFFile<ELFT>> file(
        new AArch64ELFFile<ELFT>(std::move(mb), atomizeStrings));
    if (std::error_code ec = file->parse())
      return ec;
    return std::move(file);
  }
};

template <class ELFT> class AArch64DynamicFile : public DynamicFile<ELFT> {
public:
  AArch64DynamicFile(const AArch64LinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H
