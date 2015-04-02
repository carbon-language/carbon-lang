//===- lib/ReaderWriter/ELF/AArch64/AArch64ELFReader.h --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
#define LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class AArch64LinkingContext;

typedef llvm::object::ELFType<llvm::support::little, 2, true> AArch64ELFType;

struct AArch64ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            AArch64LinkingContext &ctx) {
    return lld::elf::ELFFile<ELFT>::create(std::move(mb), ctx);
  }
};

typedef ELFObjectReader<AArch64ELFType, AArch64ELFFileCreateELFTraits,
                        AArch64LinkingContext> AArch64ELFObjectReader;

typedef ELFDSOReader<AArch64ELFType, AArch64LinkingContext> AArch64ELFDSOReader;

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
