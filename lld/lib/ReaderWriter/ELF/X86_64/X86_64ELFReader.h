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
                            X86_64LinkingContext &ctx) {
    return lld::elf::X86_64DynamicFile<ELFT>::create(std::move(mb), ctx);
  }
};

struct X86_64ELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            X86_64LinkingContext &ctx) {
    return lld::elf::X86_64ELFFile<ELFT>::create(std::move(mb), ctx);
  }
};

class X86_64ELFObjectReader
    : public ELFObjectReader<X86_64ELFType, X86_64ELFFileCreateELFTraits,
                             X86_64LinkingContext> {
public:
  X86_64ELFObjectReader(X86_64LinkingContext &ctx)
      : ELFObjectReader<X86_64ELFType, X86_64ELFFileCreateELFTraits,
                        X86_64LinkingContext>(ctx, llvm::ELF::EM_X86_64) {}
};

class X86_64ELFDSOReader
    : public ELFDSOReader<X86_64ELFType, X86_64DynamicFileCreateELFTraits,
                          X86_64LinkingContext> {
public:
  X86_64ELFDSOReader(X86_64LinkingContext &ctx)
      : ELFDSOReader<X86_64ELFType, X86_64DynamicFileCreateELFTraits,
                     X86_64LinkingContext>(ctx, llvm::ELF::EM_X86_64) {}
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_X86_64_X86_64_READER_H
