//===- lib/ReaderWriter/ELF/X86/X86ELFFile.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_X86_ELF_FILE_H
#define LLD_READER_WRITER_ELF_X86_X86_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class X86LinkingContext;

template <class ELFT> class X86ELFFile : public ELFFile<ELFT> {
public:
  X86ELFFile(std::unique_ptr<MemoryBuffer> mb, X86LinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<X86ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, X86LinkingContext &ctx) {
    return std::unique_ptr<X86ELFFile<ELFT>>(
        new X86ELFFile<ELFT>(std::move(mb), ctx));
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_X86_X86_ELF_FILE_H
