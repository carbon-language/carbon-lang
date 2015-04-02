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

namespace lld {
namespace elf {

class X86LinkingContext;
typedef llvm::object::ELFType<llvm::support::little, 2, false> X86ELFType;

struct X86ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            X86LinkingContext &ctx) {
    return lld::elf::ELFFile<ELFT>::create(std::move(mb), ctx);
  }
};

typedef ELFObjectReader<X86ELFType, X86ELFFileCreateELFTraits,
                        X86LinkingContext> X86ELFObjectReader;
typedef ELFDSOReader<X86ELFType, X86LinkingContext> X86ELFDSOReader;

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_X86_X86_ELF_READER_H
