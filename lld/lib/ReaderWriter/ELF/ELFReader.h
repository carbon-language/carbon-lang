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
#include "lld/Core/Reader.h"

namespace lld {
namespace elf {

template <typename ELFT, typename ContextT, template <typename> class FileT>
class ELFReader : public Reader {
public:
  typedef llvm::object::Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  ELFReader(ContextT &ctx) : _ctx(ctx) {}

  bool canParse(file_magic magic, const MemoryBuffer &mb) const override {
    return (FileT<ELFT>::canParse(magic) &&
            elfHeader(mb)->e_machine == ContextT::machine);
  }

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const class Registry &,
           std::vector<std::unique_ptr<File>> &result) const override {
    const Elf_Ehdr *hdr = elfHeader(*mb);
    if (auto ec = _ctx.mergeHeaderFlags(hdr->getFileClass(), hdr->e_flags))
      return ec;

    std::size_t maxAlignment =
        1ULL << llvm::countTrailingZeros(uintptr_t(mb->getBufferStart()));
    auto f = createELF<FileT>(llvm::object::getElfArchType(mb->getBuffer()),
                              maxAlignment, std::move(mb), _ctx);
    if (std::error_code ec = f.getError())
      return ec;
    result.push_back(std::move(*f));
    return std::error_code();
  }

private:
  static const Elf_Ehdr *elfHeader(const MemoryBuffer &buf) {
    return reinterpret_cast<const Elf_Ehdr *>(buf.getBuffer().data());
  }

  ContextT &_ctx;
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
