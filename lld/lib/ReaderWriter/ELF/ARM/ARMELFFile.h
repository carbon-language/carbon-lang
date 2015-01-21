//===--------- lib/ReaderWriter/ELF/ARM/ARMELFFile.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
#define LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class ARMLinkingContext;

template <class ELFT> class ARMELFFile : public ELFFile<ELFT> {
public:
  ARMELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings) {}

  static ErrorOr<std::unique_ptr<ARMELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    return std::unique_ptr<ARMELFFile<ELFT>>(
        new ARMELFFile<ELFT>(std::move(mb), atomizeStrings));
  }
};

template <class ELFT> class ARMDynamicFile : public DynamicFile<ELFT> {
public:
  ARMDynamicFile(const ARMLinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
