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

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, false> Mips32ElELFType;

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

class MipsELFObjectReader
    : public ELFObjectReader<Mips32ElELFType, MipsELFFileCreateTraits> {
  typedef ELFObjectReader<Mips32ElELFType, MipsELFFileCreateTraits>
      BaseReaderType;

public:
  MipsELFObjectReader(MipsELFFlagsMerger &flagMerger, bool atomizeStrings)
      : BaseReaderType(atomizeStrings, llvm::ELF::EM_MIPS),
        _flagMerger(flagMerger) {}

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
            std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *elfHeader(*mb);
    if (std::error_code ec = _flagMerger.merge(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::parseFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

class MipsELFDSOReader
    : public ELFDSOReader<Mips32ElELFType, MipsDynamicFileCreateELFTraits> {
  typedef ELFDSOReader<Mips32ElELFType, MipsDynamicFileCreateELFTraits>
      BaseReaderType;

public:
  MipsELFDSOReader(MipsELFFlagsMerger &flagMerger, bool useUndefines)
      : BaseReaderType(useUndefines, llvm::ELF::EM_MIPS),
        _flagMerger(flagMerger) {}

  std::error_code
  parseFile(std::unique_ptr<MemoryBuffer> mb, const Registry &registry,
            std::vector<std::unique_ptr<File>> &result) const override {
    auto &hdr = *elfHeader(*mb);
    if (std::error_code ec = _flagMerger.merge(hdr.getFileClass(), hdr.e_flags))
      return ec;
    return BaseReaderType::parseFile(std::move(mb), registry, result);
  }

private:
  MipsELFFlagsMerger &_flagMerger;
};

} // namespace elf
} // namespace lld

#endif
