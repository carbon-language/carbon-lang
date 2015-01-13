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

template <typename ELFT, typename ELFTraitsT>
class ELFObjectReader : public Reader {
public:
  typedef llvm::object::Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  ELFObjectReader(bool atomizeStrings, uint64_t machine)
      : _atomizeStrings(atomizeStrings), _machine(machine) {}

  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &buf) const override {
    return (magic == llvm::sys::fs::file_magic::elf_relocatable &&
            elfHeader(buf)->e_machine == _machine);
  }

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const class Registry &,
           std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f =
        createELF<ELFTraitsT>(llvm::object::getElfArchType(mb->getBuffer()),
                              maxAlignment, std::move(mb), _atomizeStrings);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }

  const Elf_Ehdr *elfHeader(const MemoryBuffer &buf) const {
    const uint8_t *data =
        reinterpret_cast<const uint8_t *>(buf.getBuffer().data());
    return (reinterpret_cast<const Elf_Ehdr *>(data));
  }

protected:
  bool _atomizeStrings;
  uint64_t _machine;
};

template <typename ELFT, typename ELFTraitsT>
class ELFDSOReader : public Reader {
public:
  typedef llvm::object::Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  ELFDSOReader(bool useUndefines, uint64_t machine)
      : _useUndefines(useUndefines), _machine(machine) {}

  bool canParse(file_magic magic, StringRef,
                const MemoryBuffer &buf) const override {
    return (magic == llvm::sys::fs::file_magic::elf_shared_object &&
            elfHeader(buf)->e_machine == _machine);
  }

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const class Registry &,
           std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f =
        createELF<ELFTraitsT>(llvm::object::getElfArchType(mb->getBuffer()),
                              maxAlignment, std::move(mb), _useUndefines);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }

  const Elf_Ehdr *elfHeader(const MemoryBuffer &buf) const {
    const uint8_t *data =
        reinterpret_cast<const uint8_t *>(buf.getBuffer().data());
    return (reinterpret_cast<const Elf_Ehdr *>(data));
  }

protected:
  bool _useUndefines;
  uint64_t _machine;
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
