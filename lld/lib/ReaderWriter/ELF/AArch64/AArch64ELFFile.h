//===- lib/ReaderWriter/ELF/AArch64/AArch64ELFFile.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H
#define LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class AArch64LinkingContext;

template <class ELFT> class AArch64ELFFile : public ELFFile<ELFT> {
public:
  AArch64ELFFile(StringRef name, bool atomizeStrings)
      : ELFFile<ELFT>(name, atomizeStrings) {}

  AArch64ELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings,
                 std::error_code &ec)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings, ec) {}

  static ErrorOr<std::unique_ptr<AArch64ELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    std::error_code ec;
    std::unique_ptr<AArch64ELFFile<ELFT>> file(
        new AArch64ELFFile<ELFT>(mb->getBufferIdentifier(), atomizeStrings));

    file->_objFile.reset(
        new llvm::object::ELFFile<ELFT>(mb.release()->getBuffer(), ec));

    if (ec)
      return ec;

    // Read input sections from the input file that need to be converted to
    // atoms
    if ((ec = file->createAtomizableSections()))
      return ec;

    // For mergeable strings, we would need to split the section into various
    // atoms
    if ((ec = file->createMergeableAtoms()))
      return ec;

    // Create the necessary symbols that are part of the section that we
    // created in createAtomizableSections function
    if ((ec = file->createSymbolsFromAtomizableSections()))
      return ec;

    // Create the appropriate atoms from the file
    if ((ec = file->createAtoms()))
      return ec;

    return std::move(file);
  }
};

template <class ELFT> class AArch64DynamicFile : public DynamicFile<ELFT> {
public:
  AArch64DynamicFile(const AArch64LinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_AARCH64_AARCH64_ELF_FILE_H
