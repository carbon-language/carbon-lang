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
public:
  MipsELFObjectReader(bool atomizeStrings)
      : ELFObjectReader<Mips32ElELFType, MipsELFFileCreateTraits>(
            atomizeStrings, llvm::ELF::EM_MIPS) {}
};

class MipsELFDSOReader
    : public ELFDSOReader<Mips32ElELFType, MipsDynamicFileCreateELFTraits> {
public:
  MipsELFDSOReader(bool useUndefines)
      : ELFDSOReader<Mips32ElELFType, MipsDynamicFileCreateELFTraits>(
            useUndefines, llvm::ELF::EM_MIPS) {}
};

} // namespace elf
} // namespace lld

#endif
