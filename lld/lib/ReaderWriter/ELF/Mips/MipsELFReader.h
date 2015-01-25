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

struct MipsELFFileCreateTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::MipsELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

struct MipsDynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::MipsDynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

template <class ELFT>
class MipsELFObjectReader
    : public ELFObjectReader<ELFT, MipsELFFileCreateTraits> {
  typedef ELFObjectReader<ELFT, MipsELFFileCreateTraits> BaseReaderType;

public:
  MipsELFObjectReader(MipsLinkingContext &ctx, bool atomizeStrings)
      : BaseReaderType(atomizeStrings, llvm::ELF::EM_MIPS),
        _flagMerger(ctx.getELFFlagsMerger()) {}

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
           std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *this->elfHeader(*mb);
    if (std::error_code ec = _flagMerger.merge(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::loadFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

template <class ELFT>
class MipsELFDSOReader
    : public ELFDSOReader<ELFT, MipsDynamicFileCreateELFTraits> {
  typedef ELFDSOReader<ELFT, MipsDynamicFileCreateELFTraits> BaseReaderType;

public:
  MipsELFDSOReader(MipsLinkingContext &ctx, bool useUndefines)
      : BaseReaderType(useUndefines, llvm::ELF::EM_MIPS),
        _flagMerger(ctx.getELFFlagsMerger()) {}

  std::error_code
  loadFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
           std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *this->elfHeader(*mb);
    if (std::error_code ec = _flagMerger.merge(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::loadFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

} // namespace elf
} // namespace lld

#endif
