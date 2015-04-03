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
#include "MipsELFFlagsMerger.h"
#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT>
class MipsELFObjectReader
    : public ELFObjectReader<ELFT, MipsLinkingContext, lld::elf::MipsELFFile> {
  typedef ELFObjectReader<ELFT, MipsLinkingContext, lld::elf::MipsELFFile>
      BaseReaderType;

public:
  MipsELFObjectReader(MipsLinkingContext &ctx)
      : BaseReaderType(ctx), _flagMerger(ctx.getELFFlagsMerger()) {}

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
           std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *this->elfHeader(*mb);
    if (auto ec = _flagMerger.mergeHeaderFlags(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::loadFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

template <class ELFT>
class MipsELFDSOReader : public ELFDSOReader<ELFT, MipsLinkingContext> {
  typedef ELFDSOReader<ELFT, MipsLinkingContext> BaseReaderType;

public:
  MipsELFDSOReader(MipsLinkingContext &ctx)
      : BaseReaderType(ctx), _flagMerger(ctx.getELFFlagsMerger()) {}

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
           std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *this->elfHeader(*mb);
    if (auto ec = _flagMerger.mergeHeaderFlags(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::loadFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

} // namespace elf
} // namespace lld

#endif
