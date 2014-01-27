//===- lib/ReaderWriter/ELF/PPCELFReader.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PPC_ELF_READER_H
#define LLD_READER_WRITER_PPC_ELF_READER_H

#include "ELFReader.h"
#include "PPCELFFile.h"

namespace lld {
namespace elf {

struct PPCDynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::PPCDynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

struct PPCELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::PPCELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class PPCELFObjectReader : public ELFObjectReader {
public:
  PPCELFObjectReader(bool atomizeStrings) : ELFObjectReader(atomizeStrings) {}

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const {
    error_code ec;
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<PPCELFFileCreateELFTraits>(
        llvm::object::getElfArchType(&*mb), maxAlignment, std::move(mb),
        _atomizeStrings);
    if (!f)
      return f;
    result.push_back(std::move(*f));
    return error_code::success();
  }
};

class PPCELFDSOReader : public ELFDSOReader {
public:
  PPCELFDSOReader(bool useUndefines) : ELFDSOReader(useUndefines) {}

  virtual error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<PPCDynamicFileCreateELFTraits>(
        llvm::object::getElfArchType(&*mb), maxAlignment, std::move(mb),
        _useUndefines);
    if (!f)
      return f;
    result.push_back(std::move(*f));
    return error_code::success();
  }
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
