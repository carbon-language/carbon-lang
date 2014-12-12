//===- lib/ReaderWriter/ELF/X86_64/X86_64ELFFile.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H
#define LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class X86_64LinkingContext;

template <class ELFT> class X86_64ELFFile : public ELFFile<ELFT> {
public:
  X86_64ELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings) {}

  static ErrorOr<std::unique_ptr<X86_64ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    return std::unique_ptr<X86_64ELFFile<ELFT>>(
        new X86_64ELFFile<ELFT>(std::move(mb), atomizeStrings));
  }
};

template <class ELFT> class X86_64DynamicFile : public DynamicFile<ELFT> {
public:
  X86_64DynamicFile(const X86_64LinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_X86_64_ELF_FILE_H
