//===- lib/ReaderWriter/ELF/PPCELFFile.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_PPC_ELF_FILE_H
#define LLD_READER_WRITER_ELF_PPC_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class PPCLinkingContext;

template <class ELFT> class PPCELFFile : public ELFFile<ELFT> {
public:
  PPCELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings) {}

  static ErrorOr<std::unique_ptr<PPCELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    return std::unique_ptr<PPCELFFile<ELFT>>(
        new PPCELFFile<ELFT>(std::move(mb), atomizeStrings));
  }
};

template <class ELFT> class PPCDynamicFile : public DynamicFile<ELFT> {
public:
  PPCDynamicFile(const PPCLinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_PPC_ELF_FILE_H
