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

#include "DynamicFile.h"
#include "ELFFile.h"
#include "lld/Core/File.h"
#include "lld/Core/Reader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELF.h"

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
    result.push_back(createELF(std::move(mb)));
    return std::error_code();
  }

private:
  /// Create an object depending on the runtime attributes and alignment
  /// of an ELF file.
  std::unique_ptr<File> createELF(std::unique_ptr<MemoryBuffer> mb) const {
    using namespace llvm::ELF;
    using namespace llvm::support;

    if (uintptr_t(mb->getBufferStart()) & 1)
      llvm_unreachable("Invalid alignment for ELF file!");
    unsigned char size;
    unsigned char endian;
    std::tie(size, endian) = llvm::object::getElfArchType(mb->getBuffer());
    File *file = nullptr;
    if (size == ELFCLASS32 && endian == ELFDATA2LSB) {
      file = new FileT<ELF32LE>(std::move(mb), _ctx);
    } else if (size == ELFCLASS32 && endian == ELFDATA2MSB) {
      file = new FileT<ELF32BE>(std::move(mb), _ctx);
    } else if (size == ELFCLASS64 && endian == ELFDATA2LSB) {
      file = new FileT<ELF64LE>(std::move(mb), _ctx);
    } else if (size == ELFCLASS64 && endian == ELFDATA2MSB) {
      file = new FileT<ELF64BE>(std::move(mb), _ctx);
    }
    if (!file)
      llvm_unreachable("Invalid ELF type!");
    return std::unique_ptr<File>(file);
  }

  static const Elf_Ehdr *elfHeader(const MemoryBuffer &buf) {
    return reinterpret_cast<const Elf_Ehdr *>(buf.getBuffer().data());
  }

  ContextT &_ctx;
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
