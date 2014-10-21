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

#include "AArch64ELFFile.h"
#include "ELFReader.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, true> AArch64ELFType;

struct AArch64DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::AArch64DynamicFile<ELFT>::create(std::move(mb),
                                                      useUndefines);
  }
};

struct AArch64ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::AArch64ELFFile<ELFT>::create(std::move(mb),
                                                  atomizeStrings);
  }
};

class AArch64ELFObjectReader
    : public ELFObjectReader<AArch64ELFType, AArch64ELFFileCreateELFTraits> {
public:
  AArch64ELFObjectReader(bool atomizeStrings)
      : ELFObjectReader<AArch64ELFType, AArch64ELFFileCreateELFTraits>(
            atomizeStrings, llvm::ELF::EM_AARCH64) {}
};

class AArch64ELFDSOReader
    : public ELFDSOReader<AArch64ELFType, AArch64DynamicFileCreateELFTraits> {
public:
  AArch64ELFDSOReader(bool useUndefines)
      : ELFDSOReader<AArch64ELFType, AArch64DynamicFileCreateELFTraits>(
            useUndefines, llvm::ELF::EM_AARCH64) {}
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
