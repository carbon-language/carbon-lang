//===- lib/ReaderWriter/ELF/PPC/PPCELFReader.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PPC_PPC_ELF_READER_H
#define LLD_READER_WRITER_PPC_PPC_ELF_READER_H

#include "ELFReader.h"
#include "PPCELFFile.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::big, 2, false> PPCELFType;

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

  bool canParse(file_magic magic, StringRef ext,
                const MemoryBuffer &buf) const override {
    const uint8_t *data =
        reinterpret_cast<const uint8_t *>(buf.getBuffer().data());
    const llvm::object::Elf_Ehdr_Impl<PPCELFType> *elfHeader =
        reinterpret_cast<const llvm::object::Elf_Ehdr_Impl<PPCELFType> *>(data);
    return ELFObjectReader::canParse(magic, ext, buf) &&
           elfHeader->e_machine == llvm::ELF::EM_PPC;
  }

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<PPCELFFileCreateELFTraits>(
        llvm::object::getElfArchType(mb->getBuffer()), maxAlignment,
        std::move(mb), _atomizeStrings);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }
};

class PPCELFDSOReader : public ELFDSOReader {
public:
  PPCELFDSOReader(bool useUndefines) : ELFDSOReader(useUndefines) {}

  bool canParse(file_magic magic, StringRef ext,
                const MemoryBuffer &buf) const override {
    const uint8_t *data =
        reinterpret_cast<const uint8_t *>(buf.getBuffer().data());
    const llvm::object::Elf_Ehdr_Impl<PPCELFType> *elfHeader =
        reinterpret_cast<const llvm::object::Elf_Ehdr_Impl<PPCELFType> *>(data);
    return ELFDSOReader::canParse(magic, ext, buf) &&
           elfHeader->e_machine == llvm::ELF::EM_PPC;
  }

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> &mb, const class Registry &,
            std::vector<std::unique_ptr<File>> &result) const override {
    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<PPCDynamicFileCreateELFTraits>(
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

#endif // LLD_READER_WRITER_PPC_PPC_ELF_READER_H
