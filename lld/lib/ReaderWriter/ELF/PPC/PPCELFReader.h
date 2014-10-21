//===- lib/ReaderWriter/ELF/PPC/PPCELFReader.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PPC_PPC_ELF_READER_H
#define LLD_READER_WRITER_PPC_PPC_ELF_READER_H

#include "ELFReader.h"
#include "PPCELFFile.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::big, 2, false> PPCELFType;

struct PPCDynamicFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::SharedLibraryFile>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool useUndefines) {
    return lld::elf::PPCDynamicFile<ELFT>::create(std::move(mb), useUndefines);
  }
};

struct PPCELFFileCreateELFTraits {
  typedef llvm::ErrorOr<std::unique_ptr<lld::File>> result_type;

  template <class ELFT>
  static result_type create(std::unique_ptr<llvm::MemoryBuffer> mb,
                            bool atomizeStrings) {
    return lld::elf::PPCELFFile<ELFT>::create(std::move(mb), atomizeStrings);
  }
};

class PPCELFObjectReader
    : public ELFObjectReader<PPCELFType, PPCELFFileCreateELFTraits> {
public:
  PPCELFObjectReader(bool atomizeStrings)
      : ELFObjectReader<PPCELFType, PPCELFFileCreateELFTraits>(
            atomizeStrings, llvm::ELF::EM_PPC) {}
};

class PPCELFDSOReader
    : public ELFDSOReader<PPCELFType, PPCDynamicFileCreateELFTraits> {
public:
  PPCELFDSOReader(bool useUndefines)
      : ELFDSOReader<PPCELFType, PPCDynamicFileCreateELFTraits>(
            useUndefines, llvm::ELF::EM_PPC) {}
};

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_PPC_PPC_ELF_READER_H
