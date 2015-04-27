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

template <typename FileT> class ELFReader : public Reader {
public:
  ELFReader(ELFLinkingContext &ctx) : _ctx(ctx) {}

  bool canParse(file_magic magic, MemoryBufferRef mb) const override {
    return FileT::canParse(magic);
  }

  ErrorOr<std::unique_ptr<File>>
  loadFile(std::unique_ptr<MemoryBuffer> mb,
           const class Registry &) const override {
    if (std::error_code ec = FileT::isCompatible(mb->getMemBufferRef(), _ctx))
      return ec;
    std::unique_ptr<File> ret = llvm::make_unique<FileT>(std::move(mb), _ctx);
    return std::move(ret);
  }

private:
  ELFLinkingContext &_ctx;
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
