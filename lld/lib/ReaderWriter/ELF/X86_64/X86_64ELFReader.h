//===- lib/ReaderWriter/ELF/X86_64/X86_64ELFReader.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_X86_64_X86_64_ELF_READER_H
#define LLD_READER_WRITER_X86_64_X86_64_ELF_READER_H

#include "ELFReader.h"
#include "X86_64ELFFile.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, true> X86_64ELFType;

struct X86_64DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::X86_64DynamicFile<ELFT>::create(std::move(mb),
                                                     useUndefines);
  }
};

struct X86_64ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::X86_64ELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class X86_64ELFObjectReader
    : public ELFObjectReader<X86_64ELFType, X86_64ELFFileCreateELFTraits> {
public:
  X86_64ELFObjectReader(bool atomizeStrings)
      : ELFObjectReader<X86_64ELFType, X86_64ELFFileCreateELFTraits>(
            atomizeStrings, llvm::ELF::EM_X86_64) {}
};

class X86_64ELFDSOReader
    : public ELFDSOReader<X86_64ELFType, X86_64DynamicFileCreateELFTraits> {
public:
  X86_64ELFDSOReader(bool useUndefines)
      : ELFDSOReader<X86_64ELFType, X86_64DynamicFileCreateELFTraits>(
            useUndefines, llvm::ELF::EM_X86_64) {}
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_X86_64_X86_64_READER_H
