//===- lib/ReaderWriter/ELF/X86/X86ELFReader.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_X86_X86_ELF_READER_H
#define LLD_READER_WRITER_X86_X86_ELF_READER_H

#include "ELFReader.h"
#include "X86ELFFile.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, false> X86ELFType;

struct X86DynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::X86DynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

struct X86ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::X86ELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class X86ELFObjectReader
    : public ELFObjectReader<X86ELFType, X86ELFFileCreateELFTraits> {
public:
  X86ELFObjectReader(bool atomizeStrings)
      : ELFObjectReader<X86ELFType, X86ELFFileCreateELFTraits>(
            atomizeStrings, llvm::ELF::EM_386) {}
};

class X86ELFDSOReader
    : public ELFDSOReader<X86ELFType, X86DynamicFileCreateELFTraits> {
public:
  X86ELFDSOReader(bool useUndefines)
      : ELFDSOReader<X86ELFType, X86DynamicFileCreateELFTraits>(
            useUndefines, llvm::ELF::EM_386) {}
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_X86_X86_ELF_READER_H
