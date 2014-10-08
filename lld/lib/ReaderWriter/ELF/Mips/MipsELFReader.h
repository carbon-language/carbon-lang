//===- lib/ReaderWriter/ELF/MipsELFReader.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_READER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_READER_H

#include "ELFReader.h"
#include "MipsELFFile.h"

namespace lld {
namespace elf {

struct MipsELFFileCreateTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::MipsELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class MipsELFObjectReader : public ELFObjectReader {
public:
  MipsELFObjectReader(bool atomizeStrings) : ELFObjectReader(atomizeStrings) {}

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<MipsELFFileCreateTraits>(
        llvm::object::getElfArchType(mb->getBuffer()), maxAlignment,
        std::move(mb), _atomizeStrings);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }
};

} // namespace elf
} // namespace lld

#endif
