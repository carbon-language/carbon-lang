//===- lib/ReaderWriter/ELF/AArch64/AArch64ELFReader.h --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
#define LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H

#include "AArch64ELFFile.h"
#include "ELFReader.h"

namespace lld {
namespace elf {

struct AArch64DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::AArch64DynamicFile<ELFT>::create(std::move(mb),
                                                      useUndefines);
  }
};

struct AArch64ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::AArch64ELFFile<ELFT>::create(std::move(mb),
                                                  atomizeStrings);
  }
};

class AArch64ELFObjectReader : public ELFObjectReader {
public:
  AArch64ELFObjectReader(bool atomizeStrings)
      : ELFObjectReader(atomizeStrings) {}

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<AArch64ELFFileCreateELFTraits>(
        llvm::object::getElfArchType(mb->getBuffer()), maxAlignment,
        std::move(mb), _atomizeStrings);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }
};

class AArch64ELFDSOReader : public ELFDSOReader {
public:
  AArch64ELFDSOReader(bool useUndefines) : ELFDSOReader(useUndefines) {}

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<AArch64DynamicFileCreateELFTraits>(
        llvm::object::getElfArchType(mb->getBuffer()), maxAlignment,
        std::move(mb), _useUndefines);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
