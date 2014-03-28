//===- lib/ReaderWriter/ELF/ELFReader.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_READER_H
#define LLD_READER_WRITER_ELF_READER_H

#include "CreateELF.h"
#include "DynamicFile.h"
#include "ELFFile.h"

#include "lld/ReaderWriter/Reader.h"

namespace lld {
namespace elf {

struct DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::DynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

struct ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::ELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class ELFObjectReader : public Reader {
public:
  ELFObjectReader(bool atomizeStrings) : _atomizeStrings(atomizeStrings) {}

  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &) const override {
    return (magic == llvm::sys::fs::file_magic::elf_relocatable);
  }

  error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<ELFFileCreateELFTraits>(
        llvm::object::getElfArchType(&*mb), maxAlignment, std::move(mb),
        _atomizeStrings);
    if (error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return error_code::success();
  }

protected:
  bool _atomizeStrings;
};

class ELFDSOReader : public Reader {
public:
  ELFDSOReader(bool useUndefines) : _useUndefines(useUndefines) {}

  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &) const override {
    return (magic == llvm::sys::fs::file_magic::elf_shared_object);
  }

  error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<DynamicFileCreateELFTraits>(
        llvm::object::getElfArchType(&*mb), maxAlignment, std::move(mb),
        _useUndefines);
    if (error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return error_code::success();
  }

protected:
  bool _useUndefines;
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
